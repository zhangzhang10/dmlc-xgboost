/*!
 * Copyright 2015 by Contributors
 * \file simple_dmatrix.h
 * \brief In-memory version of DMatrix.
 * \author Tianqi Chen
 */
#ifndef XGBOOST_DATA_SIMPLE_DMATRIX_H_
#define XGBOOST_DATA_SIMPLE_DMATRIX_H_

#include <xgboost/base.h>
#include <xgboost/data.h>

#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <condition_variable>


namespace xgboost {
namespace data {
// Used for single batch data.
class SimpleDMatrix : public DMatrix {
 public:
  SimpleDMatrix() = default;
  template <typename AdapterT>
  explicit SimpleDMatrix(AdapterT* adapter, float missing, int nthread);

  explicit SimpleDMatrix(dmlc::Stream* in_stream);
  ~SimpleDMatrix() override = default;

  void SaveToLocalFile(const std::string& fname);

  MetaInfo& Info() override;

  const MetaInfo& Info() const override;

  bool SingleColBlock() const override { return true; }
  DMatrix* Slice(common::Span<int32_t const> ridxs) override;

  /*! \brief magic number used to identify SimpleDMatrix binary files */
  static const int kMagic = 0xffffab01;

 private:
  BatchSet<SparsePage> GetRowBatches() override;
  BatchSet<CSCPage> GetColumnBatches() override;
  BatchSet<SortedCSCPage> GetSortedColumnBatches() override;
  BatchSet<EllpackPage> GetEllpackBatches(const BatchParam& param) override;

  MetaInfo info_;
  SparsePage sparse_page_;  // Primary storage type
  std::unique_ptr<CSCPage> column_page_;
  std::unique_ptr<SortedCSCPage> sorted_column_page_;
  std::unique_ptr<EllpackPage> ellpack_page_;
  BatchParam batch_param_;

  bool EllpackExists() const override {
    return static_cast<bool>(ellpack_page_);
  }
  bool SparsePageExists() const override {
    return true;
  }
};

class BatchedDMatrix : public DMatrix {
public:
  static BatchedDMatrix* GetBatchedDMatrix(unsigned numBatches);

  bool AddBatch(std::unique_ptr<DMatrix> batch);

  MetaInfo& Info() override { return *info_; }

  const MetaInfo& Info() const override { return *info_; }

  bool SingleColBlock() const override { return true; }

private:
  explicit BatchedDMatrix(unsigned numBatches) : nBatches_(numBatches), info_(new MetaInfo) {}

  BatchSet<SparsePage> GetRowBatches() override;

  BatchSet<CSCPage> GetColumnBatches() override {
    LOG(FATAL) << "method not implemented";
  }
  BatchSet<SortedCSCPage> GetSortedColumnBatches() override {
    LOG(FATAL) << "method not implemented";
  }
  BatchSet<EllpackPage> GetEllpackBatches(const BatchParam& param) override {
    LOG(FATAL) << "method not implemented";
  }
  DMatrix* Slice(common::Span<int32_t const> ridxs) override {
    LOG(FATAL) << "method not implemented";
  }

  bool EllpackExists() const override { return true; }
  bool SparsePageExists() const override { return false; }

  using BatchVec = std::vector<SparsePage>;
  
  class BatchSetIteratorImpl : public BatchIteratorImpl<SparsePage> {
  public:
    explicit BatchSetIteratorImpl(const BatchVec& sources)
      : sources_(sources), iter_(sources_.begin()) {}
    SparsePage& operator*() override {
      CHECK(!AtEnd());
      *iter_;
    }
    const SparsePage& operator*() const override {
      CHECK(!AtEnd());
      *iter_;
    }
    void operator++() override {
      ++iter_;
    }
    bool AtEnd() const override {
      return iter_ == sources_.end();
    }
  private:
    const BatchVec& sources_;
    BatchVec::const_iterator iter_;
  };

  static BatchedDMatrix *newMat_;
  static std::mutex batchMutex_;
  static std::condition_variable batchCv_;

  unsigned nBatches_;
  unsigned nSources_;
  BatchVec sources_;
  std::unique_ptr<MetaInfo> info_;
};
}  // namespace data
}  // namespace xgboost
#endif  // XGBOOST_DATA_SIMPLE_DMATRIX_H_
