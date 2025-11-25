/* Copyright 2020-2023 The Loimos Project Developers.
 * See the top-level LICENSE file for details.
 *
 * SPDX-License-Identifier: MIT
 */

#include "../DiseaseModel.h"
#include "../Defs.h"
#include "../loimos.decl.h"
#include "../Event.h"
#include "../readers/AttributeTable.h"
#include "gtest/gtest.h"

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace
{

  AttributeTable BuildTestAttributeTable()
  {
    AttributeTable table;
    Attribute ageAttribute;
    ageAttribute.name = "age";
    ageAttribute.dataType = DataTypes::int32_;
    Data ageDefault{};
    ageDefault.int32_val = 0;
    ageAttribute.defaultValue = ageDefault;
    table.list.push_back(ageAttribute);
    return table;
  }

  class SafeRiskyDiseaseModelTest : public ::testing::Test
  {
  protected:
    void SetUp() override
    {
      attribute_table_ = BuildTestAttributeTable();
        disease_model_.reset(new DiseaseModel(
          "../data/disease_models/safe_risky.textproto", -1.0, attribute_table_));
    }

    AttributeTable attribute_table_;
    std::unique_ptr<DiseaseModel> disease_model_;
  };

  TEST_F(SafeRiskyDiseaseModelTest, HealthyStateDependsOnAge)
  {
    const uint32_t N = 100;

    std::default_random_engine generator(12345);
    std::uniform_int_distribution<int> distribution(0, 100);

    for (int i = 0; i < N; i++)
    {
      int age = distribution(generator);
      std::vector<Data> person_data(1);
      person_data[0].int32_val = age;
      EXPECT_EQ(disease_model_->getHealthyState(person_data), age <= 50 ? 3 : 0);
    }
  }

  TEST_F(SafeRiskyDiseaseModelTest, ExposureTransitionsAreImmediate)
  {
    std::default_random_engine generator(12345);
    typedef std::tuple<int, std::tuple<int, int>> ExposureTestCase;
    const std::array<ExposureTestCase, 3> testCases = {{
        ExposureTestCase(3, std::make_tuple(4, 0)),                                // Exposure transitions are immediate
        ExposureTestCase(2, std::make_tuple(2, std::numeric_limits<Time>::max())), // Terminal states do not advance
        ExposureTestCase(4, std::make_tuple(5, 3 * DAY_LENGTH))                    // Timed transition returns duration in seconds
    }};

    for (const auto &testCase : testCases)
    {
      const int fromState = std::get<0>(testCase);
      const std::tuple<int, int> &expected = std::get<1>(testCase);
        const auto transition =
          disease_model_->transitionFromState(fromState, &generator);
      EXPECT_EQ(std::get<0>(transition), std::get<0>(expected));
      EXPECT_EQ(std::get<1>(transition), std::get<1>(expected));
    }
  }

  TEST_F(SafeRiskyDiseaseModelTest, InfectiousAndSusceptibleFlagsMatchModel)
  {
    typedef std::tuple<int, bool, bool> FlagExpectation;
    const std::array<FlagExpectation, 6> expected = {{
        FlagExpectation(0, false, true),  // healthy_safe
        FlagExpectation(1, true, false),  // infectious_safe
        FlagExpectation(2, false, false), // recovered_safe
        FlagExpectation(3, false, true),  // healthy_risky
        FlagExpectation(4, true, false),  // infectious_risky
        FlagExpectation(5, false, false)  // recovered_risky

    }};

    for (const auto &entry : expected)
    {
      const int state = std::get<0>(entry);
      const bool isInfectious = std::get<1>(entry);
      const bool isSusceptible = std::get<2>(entry);
      EXPECT_EQ(disease_model_->isInfectious(state), isInfectious);
      EXPECT_EQ(disease_model_->isSusceptible(state), isSusceptible);
    }
  }

  TEST_F(SafeRiskyDiseaseModelTest, LookupStateNamesReflectProtoOrder)
  {
    typedef std::tuple<int, std::string> NameExpectation;
    const std::array<NameExpectation, 6> expected = {{
        NameExpectation(0, "healthy_safe"),
        NameExpectation(1, "infectious_safe"),
        NameExpectation(2, "recovered_safe"),
        NameExpectation(3, "healthy_risky"),
        NameExpectation(4, "infectious_risky"),
        NameExpectation(5, "recovered_risky")
    }};

    for (const auto &entry : expected)
    {
      const int state = std::get<0>(entry);
      const std::string &name = std::get<1>(entry);
      EXPECT_EQ(disease_model_->lookupStateName(state), name);
    }
  }

  TEST(DiseaseModelPropensityTest, PropensityScalesWithDurationAndModifiers)
  {
    AttributeTable attributes = BuildTestAttributeTable();
    DiseaseModel model("../data/disease_models/covid19_onepath.textproto",
                       -1.0, attributes);

    DiseaseState susceptible_state = 0; // S_a
    DiseaseState infectious_state = 5;  // Isymp_a
    const std::array<std::pair<double, double>, 4> modifierPairs = {{
        std::make_pair(0.1, 0.1),
        std::make_pair(0.1, 1.0),
        std::make_pair(1.0, 0.1),
        std::make_pair(1.0, 1.0)
    }};

    for (const auto &modifiers : modifierPairs)
    {
      const double susceptibility_scalar = modifiers.first;
      const double infectivity_scalar = modifiers.second;
      const Time start = 0;
      const Time end = 12 * HOUR_LENGTH;

      const double expected = model.model->transmissibility() * (end - start) *
                              susceptibility_scalar * infectivity_scalar *
                              model.model->disease_states(susceptible_state).susceptibility() *
                              model.model->disease_states(infectious_state).infectivity() /
                              DAY_LENGTH;

      const double propensity = model.getPropensity(susceptible_state, infectious_state,
                                                    start, end, susceptibility_scalar, infectivity_scalar);

      EXPECT_DOUBLE_EQ(propensity, expected);
    }
  }

  TEST(DiseaseModelLogProbTest, LogProbabilityAccountsForOverlapDuration)
  {
    // N=10 is acceptable in this case

    AttributeTable attributes = BuildTestAttributeTable();
    DiseaseModel model("../data/disease_models/covid19_onepath.textproto",
                       -1.0, attributes);

    Event susceptible_event{};
    susceptible_event.personState = 0; // S_a; susceptibility 1
    susceptible_event.scheduledTime = 0;

    Event infectious_event{};
    infectious_event.personState = 5; // Isymp_a; infectivity 1
    infectious_event.scheduledTime = 6 * HOUR_LENGTH;

    double expected_base = 1.0 - model.model->transmissibility() * model.model->disease_states(0).susceptibility() * model.model->disease_states(5).infectivity();

    double expected = std::log(expected_base) * (6 * HOUR_LENGTH);

    EXPECT_DOUBLE_EQ(model.getLogProbNotInfected(susceptible_event,
                                                 infectious_event),
                     expected);
  }

} // namespace
