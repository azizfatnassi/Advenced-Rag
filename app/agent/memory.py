
import os
import json
PREFERENCES_FILE= "user_preferences.json"

def load_preferences(user_id)->list:
    if not os.path.exists(PREFERENCES_FILE) : 
        return []
    
    with open(PREFERENCES_FILE,"r") as f :
     data= json.load(f)
    return data.get(user_id,[])

def save_preferences(user_id,preference):
   
   if  os.path.exists(PREFERENCES_FILE):
      with open(PREFERENCES_FILE,"r") as f:
         data= json.load(f)
   else :
      data={}
   if user_id not in data:
      data[user_id]= []
    
   data[user_id].append(preference)

   with open (PREFERENCES_FILE,"w") as f:
      json.dump(data,f)
    
      
def get_preferences_as_text(user_id)->str:
   data=load_preferences(user_id=user_id)
   if not data:
     return  ""
   preferences_text = "\n- ".join(data)
   return f"User preferences:\n- {preferences_text}"
