import pandas as pd
import numpy as np

data = pd.read_excel('linton_supp_tableS1_S2_8Feb2020.xlsx', skiprows=[0])
deathData = pd.read_excel('linton_supp_tableS1_S2_8Feb2020.xlsx', skiprows=[0], sheet_name='TableS2')
cleanData = pd.DataFrame((data.dropna(subset=['ExposureL', 'Onset'])))
Expo = pd.Series(cleanData.ExposureL)
Onset = pd.Series(cleanData.Onset)
q1Data = pd.DataFrame({'Exposure': Expo, 'Onset': Onset})
expoLst = list(Expo)
onsetLst = list(Onset)
# 1 a

lst = []
for i in range(len(expoLst)):
    lst.append(int(str(onsetLst[i] - expoLst[i]).replace(' days 00:00:00', '')))
print(lst)
q1mean = np.mean(lst)
q1var = np.var(lst)
print("\nMean Incubation Period:", "{0:.2f}".format(q1mean))
print("Variance Incubation Period:", "{0:.2f}".format(q1var))

# 1 b
someLst = list(cleanData[cleanData['ExposureType'] != 'Lives-works-studies in Wuhan']['ExposureL'])
someOtherLst = list(cleanData[cleanData['ExposureType'] != 'Lives-works-studies in Wuhan']['Onset'])
lst = []
for i in range(len(someLst)):
    lst.append(int(str(someOtherLst[i] - someLst[i]).replace(' days 00:00:00', '')))
print(lst)
print('\nB part')
print('Mean:', np.mean(lst))
print("Variance:", np.var(lst))

cleanOnsetData = pd.DataFrame(deathData.dropna(subset=['Onset', 'Hospitalization/Isolation']))
cleanOnsetDeathData = pd.DataFrame(deathData.dropna(subset=['Onset', 'Death']))
cleanHospData = pd.DataFrame(deathData.dropna(subset=['Hospitalization/Isolation', 'Death']))
onsetLst = list(pd.Series(cleanOnsetData.Onset))
hospLst = list(pd.Series(cleanOnsetData['Hospitalization/Isolation']))
lst = []
for i in range(len(onsetLst)):
    lst.append(int(str(hospLst[i] - onsetLst[i]).replace(' days 00:00:00', '')))
print('\nC part 1')
print('Mean:', np.mean(lst))
print("Variance:", np.var(lst))

deathLst = list(pd.Series(cleanOnsetDeathData.Death))
onsetLst = list(pd.Series(cleanOnsetDeathData.Onset))
lst = []
for i in range(len(onsetLst)):
    lst.append(int(str(deathLst[i] - onsetLst[i]).replace(' days 00:00:00', '')))
print('\nC part 2')
print('Mean:', np.mean(lst))
print("Variance:", np.var(lst))

hospLst = list(pd.Series(cleanHospData['Hospitalization/Isolation']))
deathLst = list(pd.Series(cleanHospData.Death))
lst = []
for i in range(len(onsetLst)):
    lst.append(int(str(deathLst[i] - hospLst[i]).replace(' days 00:00:00', '')))
print('\nC part 3')
print('Mean:', np.mean(lst))
print("Variance:", np.var(lst))