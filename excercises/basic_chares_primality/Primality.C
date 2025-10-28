#include "primality.decl.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <vector>

CProxy_Main mainProxy;

namespace {
struct ResultRecord {
  unsigned long requestId;
  unsigned long long value;
  bool isPrime;
  bool completed;
};
}

class Main : public CBase_Main {
 public:
  Main(CkArgMsg *msg);
  void receiveResult(const PrimalityResult &result);

 private:
  void parseArgs(CkArgMsg *msg);
  void startComputation();

  unsigned long taskCount_;
  unsigned int grainSize_;
  unsigned long pendingResults_;
  std::vector<ResultRecord> results_;
};

class CheckPrimality : public CBase_CheckPrimality {
 public:
  CheckPrimality();
  void testNumber(const PrimalityTask &task);
  void testBatch(const PrimalityBatch &batch);
};

Main::Main(CkArgMsg *msg)
    : taskCount_(0), grainSize_(1), pendingResults_(0) {
  parseArgs(msg);
  mainProxy = thisProxy;
  results_.reserve(taskCount_);
  for (unsigned long id = 0; id < taskCount_; ++id) {
    results_.push_back(ResultRecord{.requestId = id, .value = 0ULL, .isPrime = false,
                                    .completed = false});
  }
  CkPrintf("[scaffold] configured for %lu tasks with grainsize %u\n", taskCount_, grainSize_);
  startComputation();
}

void Main::parseArgs(CkArgMsg *msg) {
  const unsigned long defaultTasks = 16UL;
  const unsigned int defaultGrain = 1U;
  if (msg && msg->argc > 1) {
    taskCount_ = static_cast<unsigned long>(std::strtoul(msg->argv[1], nullptr, 10));
  } else {
    taskCount_ = defaultTasks;
  }
  if (msg && msg->argc > 2) {
    grainSize_ = static_cast<unsigned int>(std::strtoul(msg->argv[2], nullptr, 10));
  } else {
    grainSize_ = defaultGrain;
  }
  grainSize_ = std::max(grainSize_, 1U);
  if (msg) {
    delete msg;
    msg = nullptr;
  }
}

void Main::startComputation() {
  pendingResults_ = taskCount_;
  CkPrintf("[scaffold] task dispatch not implemented yet; pending %lu results\n",
           pendingResults_);
  CkExit();
}

void Main::receiveResult(const PrimalityResult &result) {
  CkPrintf("[scaffold] received placeholder result for request %lu (value %llu)\n",
           result.requestId, result.value);
}

CheckPrimality::CheckPrimality() {
  CkPrintf("[scaffold] instantiated check chare %d\n", thisIndex);
}

void CheckPrimality::testNumber(const PrimalityTask &task) {
  CkPrintf("[scaffold] testNumber placeholder for request %lu\n", task.requestId);
}

void CheckPrimality::testBatch(const PrimalityBatch &batch) {
  CkPrintf("[scaffold] testBatch placeholder for first request %lu (count %u)\n",
           batch.requestIdBase, batch.count);
}

#include "primality.def.h"
