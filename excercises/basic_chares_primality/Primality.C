#include "charm++.h"
#include "primality.decl.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <vector>

CProxy_Main mainProxy;

namespace
{
struct ResultRecord
{
    unsigned long requestId;
    unsigned long long value;
    bool isPrime;
    bool completed;
};

bool isPrimeNumber(const unsigned long long number)
{
    if (number <= 1ULL)
    {
        return false;
    }
    if (number == 2ULL)
    {
        return true;
    }
    if ((number & 1ULL) == 0ULL)
    {
        return false;
    }

    const unsigned long long limit = static_cast<unsigned long long>(
        std::sqrt(static_cast<long double>(number)));
    for (unsigned long long i = 3ULL; i <= limit; i += 2ULL)
    {
        if (number % i == 0ULL)
        {
            return false;
        }
    }
    return true;
}
} // namespace

class Main : public CBase_Main
{
public:
    Main(CkArgMsg *msg);
    void receiveResult(PrimalityResultMsg *msg);

private:
    void parseArgs(CkArgMsg *msg);
    void startComputation();

    unsigned long taskCount_;
    unsigned int grainSize_;
    unsigned long pendingResults_;
    CProxy_CheckPrimality workers_;
    std::vector<ResultRecord> results_;
};

class CheckPrimality : public CBase_CheckPrimality
{
public:
    CheckPrimality();
    void testNumber(PrimalityTaskMsg *msg);
};

Main::Main(CkArgMsg *msg)
    : taskCount_(0), grainSize_(1), pendingResults_(0)
{
    parseArgs(msg);
    mainProxy = thisProxy;

    results_.reserve(taskCount_);
    for (unsigned long id = 0; id < taskCount_; ++id)
    {
        results_.push_back(ResultRecord{id, 0ULL, false, false});
    }

    startComputation();
}

void Main::parseArgs(CkArgMsg *msg)
{
    const unsigned long defaultTasks = 16UL;
    const unsigned int defaultGrain = 1U;

    if (msg && msg->argc > 1)
    {
        taskCount_ = static_cast<unsigned long>(
            std::strtoul(msg->argv[1], nullptr, 10));
    }
    else
    {
        taskCount_ = defaultTasks;
    }

    if (msg && msg->argc > 2)
    {
        grainSize_ = static_cast<unsigned int>(
            std::strtoul(msg->argv[2], nullptr, 10));
    }
    else
    {
        grainSize_ = defaultGrain;
    }

    grainSize_ = std::max(grainSize_, 1U);

    if (msg)
    {
        delete msg;
    }
}

void Main::startComputation()
{
    if (taskCount_ == 0UL)
    {
        CkPrintf("No tasks requested. Exiting.\n");
        CkExit();
        return;
    }

    static bool seeded = false;
    if (!seeded)
    {
        std::srand(static_cast<unsigned int>(std::time(nullptr)));
        seeded = true;
    }

    const int arraySize = static_cast<int>(taskCount_);
    workers_ = CProxy_CheckPrimality::ckNew(arraySize);

    pendingResults_ = taskCount_;
    for (unsigned long id = 0; id < taskCount_; ++id)
    {
        const unsigned long long value = static_cast<unsigned long long>(std::rand());
        ResultRecord &record = results_[id];
        record.value = value;
        record.isPrime = false;
        record.completed = false;

        PrimalityTaskMsg *taskMsg = new PrimalityTaskMsg;
        taskMsg->requestId = id;
        taskMsg->value = value;
    workers_[static_cast<int>(id)].testNumber(taskMsg);
    }
}

void Main::receiveResult(PrimalityResultMsg *msg)
{
    if (!msg)
    {
        return;
    }

    const unsigned long requestId = msg->requestId;
    const bool primeFlag = msg->isPrime != 0;
    delete msg;

    if (requestId >= results_.size())
    {
        CkAbort("Received result with invalid requestId");
        return;
    }

    ResultRecord &record = results_[requestId];
    record.isPrime = primeFlag;
    record.completed = true;

    if (pendingResults_ == 0UL)
    {
        CkAbort("Pending result counter underflow");
        return;
    }

    --pendingResults_;
    if (pendingResults_ == 0UL)
    {
        CkPrintf("Primality results (value, isPrime):\n");
        for (const ResultRecord &r : results_)
        {
            CkPrintf("  %12llu -> %s\n", r.value, r.isPrime ? "true" : "false");
        }
        CkExit();
    }
}

CheckPrimality::CheckPrimality() = default;

void CheckPrimality::testNumber(PrimalityTaskMsg *msg)
{
    if (!msg)
    {
        return;
    }

    const unsigned long requestId = msg->requestId;
    const unsigned long long value = msg->value;
    delete msg;

    PrimalityResultMsg *result = new PrimalityResultMsg;
    result->requestId = requestId;
    result->isPrime = isPrimeNumber(value) ? 1 : 0;
    mainProxy.receiveResult(result);
}

#include "primality.def.h"
