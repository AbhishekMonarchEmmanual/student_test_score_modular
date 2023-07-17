import streamlit as st
import pandas as pd 
import os

class Prediction_input:
        
    def my_title_page(self):
        """_
        It is the program just for creating the webpage 
        from where i can take inputs from web and predict the vaue 
        based on my trained model
        run input webpage seprately on app.py
        and run main.py seprately. Also first run app.py and create a df will be the best option
        as i am yet to handle many exceptions with time will do it for now it is what is .  
        it will create the df that you want to predict the values after 
        training and creating our model 
        """
        
        st.title("LETS TRY TO PREDICT AND USE OUR MODEL :umbrella_with_rain_drops:")
        gender = st.text_input("INPUT gender ex: [male , female]")
        race_ethnicity= st.text_input("INPUT race_ethnicity ex: ['group B', 'group C', 'group A', 'group D', 'group E']")
        parental_level_of_education= st.text_input("INPUT parental_level_of_education ex: [bachelor's degree, some college, master's degree,associate's degree, high school, some high school]")
        lunch= st.text_input("INPUT lunch ex: ['standard', 'free/reduced']")
        test_preparation_course= st.text_input("INPUT test_preparation_course ex: ['none', 'completed']")
        reading_score= st.text_input("INPUT reading_score ex: [0-100]")
        writing_score= st.text_input("INPUT writing_score ex; [0-100]")
        if st.checkbox("Create Dictionary", key= 'create dictionary'):
            data_dict={'gender' : gender, 'race_ethnicity' : race_ethnicity,
                      'parental_level_of_education': parental_level_of_education,
                      'lunch' : lunch,
                      'test_preparation_course' : test_preparation_course,
                      'reading_score' : reading_score,
                      'writing_score': writing_score
                      }
            st.write(data_dict)
            if st.checkbox("wanna see df"):
                df = pd.DataFrame([data_dict])
                st.write(df)
                if st.checkbox("want df to predict value"):
                    os.makedirs("predict_df" ,exist_ok=True)
                    df.to_csv("predict_df/df.csv", index= False)
                    return df
if __name__ == "__main__":
    webpage  = Prediction_input()                
    title_input_page = webpage.my_title_page()











    
    
    
