import pandas as pd



def analyzeHorizonImpact(csvDir):
    # Load sequential Rollout Data
    horizonVals = [4,10,20]
    startVals = [0,50,100]
    endVals = [50,100,150]
    std = 3

    dfCompare = pd.DataFrame(columns=["Seed"])
    for horizon in horizonVals:
        dfTotal_Hori = pd.DataFrame()
        for ind in zip(startVals,endVals):
            csvFileName_R = f"BPFalse_R_{ind[0]}_to_{ind[1]}_STD_{std}_H_{horizon}_T_0.5_cem_20_NN_50_5_REWARD.csv"
            csvFileName_S = f"BPFalse_R_{ind[0]}_to_{ind[1]}_STD_{std}_H_{horizon}_T_0.5_cem_20_NN_50_5_STEPS.csv"
            dfReward = pd.read_csv(csvDir+csvFileName_R)
            dfStep = pd.read_csv(csvDir+csvFileName_S)
            

            dfRewardV2 = dfReward[dfReward.columns[0:2]]
            dfStepV2 = dfStep[dfStep.columns[0:2]]
            dfRewardV2.columns = ['Seed', 'Reward']
            dfStepV2.columns = ['Seed', 'Steps']
            dfRewardV2.loc[:, "Seed"] += ind[0]
            dfStepV2.loc[:, "Seed"] += ind[0]


            dfFin = dfRewardV2.merge(dfStepV2,on="Seed",how="left")
            

            dfTotal_Hori = pd.concat([dfTotal_Hori,dfFin])

        dfTotal_Hori.columns = ["Seed",f"{horizon}_Reward",f"{horizon}_Steps"]

        dfCompare = dfCompare.merge(dfTotal_Hori,how="outer")

    # Load Base Policy Data
    dfTotal_BP = pd.DataFrame()
    for ind in zip(startVals,endVals): 
        BP_csvFileName_S = f"BPTrue_R_{ind[0]}_to_{ind[1]}_STEPS.csv"
        BP_csvFileName_R = f"BPTrue_R_{ind[0]}_to_{ind[1]}_REWARD.csv"
        dfReward_BP = pd.read_csv(csvDir+BP_csvFileName_R)
        dfStep_BP = pd.read_csv(csvDir+BP_csvFileName_S)

        dfReward_BPV2 = dfReward_BP[dfReward_BP.columns[0:2]]
        dfStep_BPV2 = dfStep_BP[dfStep_BP.columns[0:2]]
        dfReward_BPV2.columns = ['Seed', 'Reward_BP']
        dfStep_BPV2.columns = ['Seed', 'Steps_BP']
        dfReward_BPV2.loc[:,"Seed"] += ind[0]
        dfStep_BPV2.loc[:,"Seed"] += ind[0]

        dfFin_BP = dfReward_BPV2.merge(dfStep_BPV2,on="Seed",how="left")
        dfTotal_BP = pd.concat([dfTotal_BP,dfFin_BP])

    dfCompare = dfCompare.merge(dfTotal_BP,how="outer")


    import ipdb; ipdb.set_trace()

        


            







if __name__ == "__main__":
    csvDir = "/Users/athmajanvivekananthan/WCE/JEPA - MARL/tdmpc_mpe/csv_results/"
    analyzeHorizonImpact(csvDir)