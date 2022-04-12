//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#ifndef SRC_RAISIMGYMVECENV_HPP
#define SRC_RAISIMGYMVECENV_HPP

#include "RaisimGymEnv.hpp"
#include "omp.h"
#include "Yaml.hpp"

namespace raisim {

template<class ChildEnvironment>
class VectorizedEnvironment {

 public:

  explicit VectorizedEnvironment(std::string resourceDir, std::string cfg)
      : resourceDir_(resourceDir) {
    Yaml::Parse(cfg_, cfg);
	  raisim::World::setActivationKey("/home/opensim2020/raisim_v5_workspace/raisimLib/rsc/activation.raisim");
    if(&cfg_["render"])
      render_ = cfg_["render"].template As<bool>();
  }

  ~VectorizedEnvironment() {
    for (auto *ptr: environments_)
      delete ptr;
  }

  void init() {
    omp_set_num_threads(cfg_["num_threads"].template As<int>());
    num_envs_ = cfg_["num_envs"].template As<int>();

    for (int i = 0; i < num_envs_; i++) {
      environments_.push_back(new ChildEnvironment(resourceDir_, cfg_, render_ && i == 0));
      environments_.back()->setSimulationTimeStep(cfg_["simulation_dt"].template As<double>());
      environments_.back()->setControlTimeStep(cfg_["control_dt"].template As<double>());
    }

    setSeed(0);

    for (int i = 0; i < num_envs_; i++) {
      // only the first environment is visualized
      environments_[i]->init();
      environments_[i]->reset();
    }

    obDim_ = environments_[0]->getObDim();
    actionDim_ = environments_[0]->getActionDim();
    RSFATAL_IF(obDim_ == 0 || actionDim_ == 0, "Observation/Action dimension must be defined in the constructor of each environment!")

    /// generate reward names  by skoo
    /// compute it once to get reward names. actual value is not used
    environments_[0]->updateExtraInfo();
    for (auto &re: environments_[0]->extraInfo_)
      extraInfoName_.push_back(re.first);
  }

  // by skoo
  std::vector<std::string> &getExtraInfoNames() {
    return extraInfoName_;
  }

  // resets all environments and returns observation
  void reset() {
    for (auto env: environments_)
      env->reset();
  }

  void observe(Eigen::Ref<EigenRowMajorMat> &ob) {
#pragma omp parallel for
    for (int i = 0; i < num_envs_; i++)
      environments_[i]->observe(ob.row(i));
  }

  void step(Eigen::Ref<EigenRowMajorMat> &action,
            Eigen::Ref<EigenVec> &reward,
            Eigen::Ref<EigenBoolVec> &done,
            Eigen::Ref<EigenRowMajorMat> &extraInfo) {
#pragma omp parallel for
    for (int i = 0; i < num_envs_; i++)
      perAgentStep(i, action, reward, done, extraInfo);
  }

  void turnOnVisualization() { if(render_) environments_[0]->turnOnVisualization(); }
  void turnOffVisualization() { if(render_) environments_[0]->turnOffVisualization(); }
  void startRecordingVideo(const std::string& videoName) { if(render_) environments_[0]->startRecordingVideo(videoName); }
  void stopRecordingVideo() { if(render_) environments_[0]->stopRecordingVideo(); }

  void setSeed(int seed) {
    int seed_inc = seed;
    for (auto *env: environments_)
      env->setSeed(seed_inc++);
  }

  void close() {
    for (auto *env: environments_)
      env->close();
  }

  void isTerminalState(Eigen::Ref<EigenBoolVec>& terminalState) {
    for (int i = 0; i < num_envs_; i++) {
      float terminalReward;
      terminalState[i] = environments_[i]->isTerminalState(terminalReward);
    }
  }

  void setSimulationTimeStep(double dt) {
    for (auto *env: environments_)
      env->setSimulationTimeStep(dt);
  }

  void setControlTimeStep(double dt) {
    for (auto *env: environments_)
      env->setControlTimeStep(dt);
  }

  int getObDim() { return obDim_; }
  int getActionDim() { return actionDim_; }
  int getNumOfEnvs() { return num_envs_; }
  int getExtraInfoDim() { return extraInfoName_.size(); }

  ////// optional methods //////
  void curriculumUpdate() {
    for (auto *env: environments_)
      env->curriculumUpdate();
  };

 private:

  inline void perAgentStep(int agentId,
                           Eigen::Ref<EigenRowMajorMat> &action,
                           Eigen::Ref<EigenVec> &reward,
                           Eigen::Ref<EigenBoolVec> &done,
                           Eigen::Ref<EigenRowMajorMat> &extraInfo) {   // by skoo
    reward[agentId] = environments_[agentId]->step(action.row(agentId));

    float terminalReward = 0;
    done[agentId] = environments_[agentId]->isTerminalState(terminalReward);

    // by skoo
    environments_[agentId]->updateExtraInfo();
    for (int j = 0; j < extraInfoName_.size(); j++)
      extraInfo(agentId, j) = environments_[agentId]->extraInfo_[extraInfoName_[j]];

    if (done[agentId]) {
      environments_[agentId]->reset();
      reward[agentId] += terminalReward;
    }
  }

  std::vector<ChildEnvironment *> environments_;
  std::vector<std::string> extraInfoName_;   // by skoo
  int num_envs_ = 1;
  int obDim_ = 0, actionDim_ = 0;
  bool recordVideo_=false, render_=false;
  std::string resourceDir_;
  Yaml::Node cfg_;
};

}

#endif //SRC_RAISIMGYMVECENV_HPP
