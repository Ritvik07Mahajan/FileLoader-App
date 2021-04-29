# Core Pkgs
import streamlit as st 
import os
import base64

# Time Pkg
import time
timestr = time.strftime("%Y%m%d-%H%M%S")

# Templates
file_name = 'yourdocument' + timestr + '.txt'
HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""


# EDA Pkgs
import pandas as pd 
import numpy as np 
import time,json

# Data Viz Pkg
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use("Agg")
import seaborn as sns 


# ML Packages
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# ML Packages For Vectorization of Text For Feature Extraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def writetofile(text,file_name):
	with open(os.path.join('downloads',file_name),'w') as f:
		f.write(text)

def make_downloadable(filename):
	readfile = open(os.path.join("downloads",filename)).read()
	b64 = base64.b64encode(readfile.encode()).decode()
	href = '<a href="data:file/readfile;base64,{}">Download File File</a> (right-click and save as &lt;some_name&gt;.txt)'.format(b64)
	return href



def main():
	"""Semi Automated ML App with Streamlit """

	activities = ["EDA","Plots","About"]	
	choice = st.sidebar.selectbox("Select Activities",activities)

	if choice == 'EDA':
		st.subheader("Exploratory Data Analysis")

		data = st.file_uploader("Upload a Dataset", type=["csv", "txt"])
		if data is not None:
			data.seek(0)
			df = pd.read_csv(data)
			st.dataframe(df.head())

			if st.checkbox("Show Shape"):
				st.write(df.shape)

			if st.checkbox("Show Columns"):
				all_columns = df.columns.to_list()
				st.write(all_columns)

			if st.checkbox("Summary"):
				st.write(df.describe())

			if st.checkbox("Show Selected Columns"):
				selected_columns = st.multiselect("Select Columns",all_columns)
				new_df = df[selected_columns]
				st.dataframe(new_df)

			if st.checkbox("Show Value Counts"):
				st.write(df.iloc[:,-1].value_counts())

			if st.checkbox("Correlation Plot(Matplotlib)"):
				st.set_option('deprecation.showPyplotGlobalUse', False)
				plt.matshow(df.corr())
				st.pyplot()

			if st.checkbox("Correlation Plot(Seaborn)"):
				st.write(sns.heatmap(df.corr(),annot=True))
				st.pyplot()


			if st.checkbox("Pie Plot"):

				all_columns = df.columns.to_list()
				column_to_plot = st.selectbox("Select 1 Column",all_columns)
				pie_plot = df[column_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
				st.write(pie_plot)
				st.pyplot()



	elif choice == 'Plots':
		
		st.subheader("Data Visualization")
		data = st.file_uploader("Upload a Dataset", type=["csv", "txt"])
		if data is not None:
			data.seek(0)
			df = pd.read_csv(data)
			st.dataframe(df.head())


			if st.checkbox("Show Value Counts"):
				st.write(df.iloc[:,-1].value_counts().plot(kind='bar'))
				st.pyplot()
		
			# Customizable Plot

			all_columns_names = df.columns.tolist()
			type_of_plot = st.selectbox("Select Type of Plot",["area","bar","line","hist","box","kde"])
			selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names)

			if st.button("Generate Plot"):
				st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))

				# Plot By Streamlit
				if type_of_plot == 'area':
					cust_data = df[selected_columns_names]
					st.area_chart(cust_data)

				elif type_of_plot == 'bar':
					cust_data = df[selected_columns_names]
					st.bar_chart(cust_data)

				elif type_of_plot == 'line':
					cust_data = df[selected_columns_names]
					st.line_chart(cust_data)

				# Custom Plot 
				elif type_of_plot:
					cust_plot= df[selected_columns_names].plot(kind=type_of_plot)
					st.write(cust_plot)
					st.pyplot()



	elif choice == 'About':
		st.subheader("About")
		st.text("Made By Ritvik Mahajan")
		st.text("Student of Government college of Engineering and technoloy Jammu")


if __name__ == '__main__':
	main()