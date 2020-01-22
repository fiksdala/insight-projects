#%%
import pandas as pd
import pickle

#%%
cahps = pickle.load(open('data/interim/cahps.pickle', 'rb'))
provider = pickle.load(open('data/interim/provider.pickle', 'rb'))
provider_general = pickle.load(
    open('data/interim/provider_general.pickle', 'rb')
)
puf = pickle.load(open('data/interim/puf.pickle', 'rb'))

#%%
# cahps: all but EMO_REL_MBV have same number of obs, so keep the rest
cahps_dict = {
'TEAM_COMM_TBV'	'The hospice team always communicated well'
'TIMELY_CARE_BBV'	The hospice team sometimes or never provided timely help
'EMO_REL_MBV'
'SYMPTOMS_BBV'	The patient sometimes or never got the help they needed for pain and symptoms
'RATING_BBV'	Caregivers rated the hospice agency a 6 or lower
'TEAM_COMM_MBV'	The hospice team usually  communicated well
'RESPECT_TBV'	The hospice team always treated the patient with respect
'RECOMMEND_TBV'	YES, they would definitely recommend the hospice
EMO_REL_TBV	The hospice team provided the right amount of emotional and spiritual support
SYMPTOMS_MBV	The patient usually got the help they needed for pain and symptoms
TEAM_COMM_BBV	The hospice team sometimes or never communicated well
TRAINING_BBV	They did not receive the training they needed
RECOMMEND_MBV	YES, they would probably recommend the hospice
SYMPTOMS_TBV	The patient always got the help they needed for pain and symptoms
RESPECT_MBV	The hospice team usually treated the patient with respect
TIMELY_CARE_TBV	The hospice team always provided timely help
RATING_TBV	Caregivers rated the hospice agency a 9 or 10
TRAINING_MBV	They somewhat received the training they needed
EMO_REL_BBV	The hospice team did |not| provide the right amount of emotional and spiritual support
RECOMMEND_BBV	NO, they would probably not or definitely not recommend the hospice
RESPECT_BBV	The hospice team sometimes or never treated the patient with respect
RATING_MBV	Caregivers rated the hospice agency a 7 or 8
TIMELY_CARE_MBV	The hospice team usually provided timely help
TRAINING_TBV	They definitely received the training they needed
}