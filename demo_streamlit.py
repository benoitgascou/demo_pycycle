import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from streamlit_folium import folium_static
import folium
from sklearn.cluster import KMeans

import scipy.stats as stats
from sklearn import model_selection, preprocessing
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression, SelectFromModel
from sklearn.model_selection import train_test_split
from datetime import timedelta


from PIL import Image
image = Image.open('velo.jpg')
st.image(image, use_column_width=True, width=500)

df_acc = pd.read_csv('df_acc.csv', sep=';')
#concaténation des 4 fichiers sources du NB, déjà retraité des NA, filtré sur la période sept 2019 à déc 2019 et accidents vélo

df_ml = pd.read_csv('df_ml.csv', sep=';')
#fichier déjà retraité des NA, avec les variables numériques déjà ajoutées pour optimiser le temps de traitement

plan_df = pd.read_csv('plan_df.csv', sep=';')
plan_df_jour = pd.read_csv('plan_df_jour.csv', sep=';')
plan_df_nuit = pd.read_csv('plan_df_nuit.csv', sep=';')
plan_df_2019 = pd.read_csv('plan_df_2019.csv', sep=';')
#fichier sources cartographies, flitrés par sites, doublons url photos déjà traités


dates = pd.read_csv('dates.csv', sep=';')

######################
###Plan de l'appli####
######################

###Data Visualisation

##Dataviz Trafic
dataviz1 = "Analyse du trafic cycliste à Paris"

dataviz1_1 = "Cartographie des sites de comptage"
dataviz1_2 = "Evolution du trafic entre sep 2019 et déc 2020"


##Dataviz Accidents
dataviz2 = "Analyse des accidents cyclistes à Paris"

dataviz2_1 = "Cartographie des accidents"
dataviz2_2 = "Statistiques concernant les personnes accidentées"

###Data Visualisation
machine_learning1 = "Prédictions des comptages sur les derniers jours du mois"
machine_learning2 = "Prédictions des comptages sur les derniers mois de la période"


st.sidebar.title("Plan de l'application")
st.sidebar.markdown(body="<hr>", unsafe_allow_html=True)

select_theme = st.sidebar.radio("Sélectionnez un thème :", ('Data Visualisation', 'Machine Learning'))
st.sidebar.markdown(body="<hr>", unsafe_allow_html=True)

######################
####   DATAVIZ    ####
######################
if select_theme == 'Data Visualisation':

	st.sidebar.header(select_theme)
	select_dataviz = st.sidebar.radio(
	"Sélectionnez une Dataviz :", (dataviz1_1, dataviz1_2, dataviz2_1, dataviz2_2))
	st.sidebar.markdown(body="<hr>", unsafe_allow_html=True)


#DATAVIZ TRAFIC
###################################
###################################

	# if select_dataviz == dataviz1:
	# 	st.sidebar.subheader(dataviz1)
	# 	select_traf = st.sidebar.radio("", (dataviz1_1, dataviz1_2))



#Cartographie des sites de comptage
###################################
	if select_dataviz == dataviz1_1:
		st.title(dataviz1)
		st.header(select_dataviz)

		st.sidebar.subheader(select_dataviz)
		periode = st.sidebar.radio('Sélectionnez les éléments à afficher sur la carte :', ("Trafic sur 24h", "Trafic Jour", "Trafic Nuit" ))


		if periode == "Trafic Jour":
			plan_df = plan_df_jour
		elif periode == "Trafic Nuit":
			plan_df = plan_df_nuit
				
		if st.sidebar.checkbox("Clustering", value=False):
			#df_cluster = plan_df.[['Identifiant du site de comptage','lat','long']]
			kmeans = KMeans(n_clusters = 3)
			kmeans.fit(plan_df[["Comptage horaire", "lat", "long"]])
			centroids = kmeans.cluster_centers_
			labels = kmeans.labels_
			#Ajouter les labels au DF "df_cluster"
			plan_df["groupe"] = pd.DataFrame(labels)

			#Affichage carte avec clustering
			carte = folium.Map(location = [48.86, 2.341886], zoom_start = 12, min_zoom=12)
			for nom, comptage, latitude, longitude, image, groupe in zip(plan_df["Nom du site de comptage"],
					                                                     plan_df["Comptage horaire"],
					                                                     plan_df["lat"],
					                                                     plan_df["long"],
					                                                     plan_df["Lien vers photo du site de comptage"],
					                                                     plan_df["groupe"]):
			    if groupe == 0:
			        couleur = "#d9152a"
			    elif groupe == 1:
			        couleur = "#368fe5"
			    elif groupe == 2:
			        couleur = "#129012"
			    else :
			    	couleur ="#368fe5"
			    
			    
			    pp = "<strong>" + nom + "</strong>" + "<br>Comptage horaire : " + str(round(comptage,2)) + "<br><img src='" + image + "', width=100%>"
			    folium.CircleMarker(location=[latitude, longitude],
			                        radius=comptage/10,
			                        popup = folium.Popup(pp, max_width = 300),
			                        tooltip= "<strong>" + nom + "</strong>",
			                        color = couleur,
			                        fill_color = couleur,
			                        fill_opacity=0.3
			                       ).add_to(carte)
			folium_static(carte)


		else:
			couleur = "#368fe5"
			#Affichage carte sans clustering
			carte = folium.Map(location = [48.86, 2.341886], zoom_start = 12, min_zoom=12)
			for nom, comptage, latitude, longitude, image in zip(plan_df["Nom du site de comptage"],
					                                                     plan_df["Comptage horaire"],
					                                                     plan_df["lat"],
					                                                     plan_df["long"],
					                                                     plan_df["Lien vers photo du site de comptage"]):
			    
			    pp = "<strong>" + nom + "</strong>" + "<br>Comptage horaire : " + str(round(comptage,2)) + "<br><img src='" + image + "', width=100%>"
			    folium.CircleMarker(location=[latitude, longitude],
			                        radius=comptage/10,
			                        popup = folium.Popup(pp, max_width = 300),
			                        tooltip= "<strong>" + nom + "</strong>",
			                        color = couleur,
			                        fill_color = couleur,
			                        fill_opacity=0.3
			                       ).add_to(carte)
			folium_static(carte)

		#commentaires carte		
		# st.markdown(body="<ul>"
		# 	             "<li>69 sites de comptage</li>"
		# 	             "<li>Maillage faible : 3,5 sites / arrondissement<br>"
		# 	             "Quais de Seine & axe Nord-Sud +++<br>"
		# 	             "Axes Est-Ouest --</li>"
		# 	             "<li>Sites classés en 3 couleurs selon l’intensité du trafic (modèle de clustering K-means)<br>"
		# 	             "<font size=5>o</font>   Hypercentre, 10e, 11e<br>"
		# 	             "<font size=4>o</font>   Quais, Batignolles, La Villette, 14e, 15e<br>"
		# 	             "<font size=2>o</font>   Périphérie, 6e, 7e, 8e, 13e"
		# 	             "</li></ul>",
		# 	             unsafe_allow_html=True)





#Evolution du trafic de sept 2019 à déc 2020
############################################
	if select_dataviz == dataviz1_2:
		st.sidebar.subheader(select_dataviz)
		st.title(dataviz1)
		st.header(select_dataviz)
		dates['Date'] = pd.to_datetime(dates['Date'], errors='coerce')
		dates.set_index('Date',inplace=True)
		fig = plt.figure(figsize = (20, 10))
		st.line_chart(dates)




#DATAVIZ ACCIDENTS
###################################
###################################


	# if select_dataviz == dataviz2:
	# 	st.sidebar.subheader(dataviz2)
	# 	select_acc = st.sidebar.radio("", (dataviz2_1, dataviz2_2))	


#Cartographie accidents
#######################
	if select_dataviz == dataviz2_1:
		st.subheader(dataviz2_1)
		st.title(dataviz2)
		st.header(select_dataviz)

		df_acc["lat"] = df_acc["lat"].astype(float)
		df_acc["long"] = df_acc["long"].astype(float)

		st.sidebar.subheader(select_dataviz)
		st.sidebar.markdown('Sélectionnez les éléments à afficher sur la carte :')
		legers = st.sidebar.checkbox("Accidents avec cycliste indemne ou blessé léger", value=False)
		hosp = st.sidebar.checkbox("Accidents avec cycliste blessé et hospitalisé", value=False)
		tues = st.sidebar.checkbox("Accidents avec cycliste tué", value=False)
		trafic = st.sidebar.checkbox("Sites de comptages", value=False)
		pcycl = st.sidebar.checkbox("Accidents survenus sur piste cyclable", value=False)
		hors_pcycl = st.sidebar.checkbox("Accidents survenus hors piste cyclable", value=False)

		# affichage des sites de comptage et des accidents impliquant un vélo sur la carte

		carte = folium.Map(location = [48.86, 2.341886], zoom_start = 12, min_zoom=12)



		#Affichages des accidents
		if legers:
			for latitude, longitude, adr, situ in zip(df_acc[(df_acc['grav'] == 1) | (df_acc['grav'] == 4)]["lat"],
				                                      df_acc[(df_acc['grav'] == 1) | (df_acc['grav'] == 4)]["long"],
				                                      df_acc[(df_acc['grav'] == 1) | (df_acc['grav'] == 4)]["adr"],
				                                      df_acc[(df_acc['grav'] == 1) | (df_acc['grav'] == 4)]["situ"]):
			    folium.CircleMarker(location=[latitude, longitude],
			                        radius=4,
			                        tooltip= "<strong>" + adr + "</strong>",
			                        color = "green",
			                        fill_color = "green",
			                        fill_opacity=1
			                       ).add_to(carte)

		if hosp:
			for latitude, longitude, adr, situ in zip(df_acc[(df_acc['grav'] == 3)]["lat"],
				                                      df_acc[(df_acc['grav'] == 3)]["long"],
				                                      df_acc[(df_acc['grav'] == 3)]["adr"],
				                                      df_acc[(df_acc['grav'] == 3)]["situ"]):
			    folium.CircleMarker(location=[latitude, longitude],
				                    radius=4,
				                    tooltip= "<strong>" + adr + "</strong>",
				                    color = "orange",
				                    fill_color = "orange",
				                    fill_opacity=1
				                   ).add_to(carte)

		if tues:
			for latitude, longitude, adr, situ in zip(df_acc[(df_acc['grav'] == 2)]["lat"],
				                                      df_acc[(df_acc['grav'] == 2)]["long"],
				                                      df_acc[(df_acc['grav'] == 2)]["adr"],
				                                      df_acc[(df_acc['grav'] == 2)]["situ"]):
			    folium.CircleMarker(location=[latitude, longitude],
				                    radius=4,
				                    tooltip= "<strong>" + adr + "</strong>",
				                    color = "red",
				                    fill_color = "red",
				                    fill_opacity=1
				                   ).add_to(carte)

		if pcycl:
			for latitude, longitude in zip(df_acc[(df_acc['situ'] == 5)]["lat"], df_acc[(df_acc['situ'] == 5)]["long"]):
			    folium.CircleMarker(location=[latitude, longitude],
			                        radius=4,
			                        color = "yellow",
			                        fill_color = None,
			                        fill_opacity=0
			                       ).add_to(carte)

		if hors_pcycl:
			for latitude, longitude in zip(df_acc[(df_acc['situ'] != 5)]["lat"], df_acc[(df_acc['situ'] != 5)]["long"]):
				folium.CircleMarker(location=[latitude, longitude],
				                    radius=4,
				                    color = "black",
				                    fill_color = None,
				                    fill_opacity=0
				                   ).add_to(carte)


		#Affichage des sites de comptage
		if trafic:
			for nom, comptage, latitude, longitude, image in zip(plan_df_2019["Nom du site de comptage"],
			                                                     plan_df_2019["Comptage horaire"],
			                                                     plan_df_2019["lat"],
			                                                     plan_df_2019["long"],
			                                                     plan_df_2019["Lien vers photo du site de comptage"]):
				pp = "<strong>" + nom + "</strong>" + "<br>Comptage horaire : " + str(round(comptage,2)) + "<br><img src='" + image + "', width=100%>"
				folium.CircleMarker(location=[latitude, longitude],radius=comptage/8,popup = folium.Popup(pp, max_width = 300),tooltip= "<strong>" + nom + "</strong>",color = "#368fe5",fill_color = "#368fe5",fill_opacity=0.3).add_to(carte)
		folium_static(carte)

		#commentaires carte		
		st.markdown(body=
		#"Données sep - déc 2019 : <strong>644 accidents</strong> impliquant des vélos & ayant entraîné des blessures<br>"
		"<strong><font size=20><span style='color:green'>.</span></font></strong>   accident avec blessé léger  - "
		"<strong><font size=20><span style='color:orange'>.</span></font></strong>   accident avec blessé hospitalisé  - "
		"<strong><font size=20><span style='color:red'>.</span></font></strong>   accident avec décés<br>"
		"<strong><font size=5><span style='color:yellow'>o</span></font></strong>  accident sur piste cyclable  - "
		"<strong><font size=5><span style='color:black'>o</span></font></strong>  accident hors piste cyclable", unsafe_allow_html=True)


#Stats cyclistes accidentés
###########################
	if select_dataviz == dataviz2_2:
		st.title(dataviz2)
		st.header(select_dataviz)

		#intitulés de la liste de choix :
		diag1 = "du sexe / du niveau de gravité"
		diag2 = "de l'âge"
		diag3 = "du trajet"
		diag4 = "du type de voie"
		diag5 = "de la météo"

		st.sidebar.subheader(select_dataviz)
		select_diag = st.sidebar.radio("Cyclistes accidentés en fonction :", (diag1, diag2, diag3, diag4, diag5))

		#accidentés par sexe
		####################
		if select_diag == diag1:


			#graphe sexe
			fig = plt.figure(figsize=(8, 8))
			plt.title("Cyclistes accidentés selon le sexe", fontsize=12)
			df_sex = df_acc.groupby('sexe', as_index = False).agg({'sexe':'count'})
			plt.pie(x = df_sex.sexe,
			        labels = ['Homme', 'Femme'],
			        explode = [0, 0.1],
			        autopct = lambda x : str(round(x, 2)) + '%',
			        pctdistance = 0.7,
			        labeldistance = 1.1,
			        shadow = True,
			        radius= 0.8);
			st.pyplot(fig)
			st.markdown("3/4 des accidentés à vélo sont des hommes contre 1/4 de femmes. Le boom des livraisons à"
			"vélo, métier majoritairement masculin, joue sur ces chiffres. Comparons la gravité des "
			"accidents en fonction du sexe :")


			#graphe hommes et gravité
			fig = plt.figure(figsize=(8, 8))
			plt.title("Gravité pour les hommes", fontsize=15)
			df_hom = df_acc[df_acc['sexe'] == 1]
			df_hom = df_hom.groupby('grav', as_index = False).agg({'grav':'count'})
			plt.pie(x = df_hom.grav,
			        labels = ['Indemne ', 'Tué ', 'Blessé hospitalisé', 'Blessé léger'],
			        explode = [0, 0.2, 0.2, 0],
			        autopct = lambda x : str(round(x, 2)) + '%',
			        pctdistance = 0.7,
			        labeldistance = 1.1,
			        shadow = True,
			        colors = ("lightgreen", "red", "orange", "lightyellow"))
			st.pyplot(fig)


			#graphe femmes et gravité		
			fig = plt.figure(figsize=(8, 8))
			plt.title("Gravité pour les femmes", fontsize=15)
			df_fem = df_acc[df_acc['sexe'] == 2]
			df_fem = df_fem.groupby('grav', as_index = False).agg({'grav':'count'})
			plt.pie(x = df_fem.grav,
			        labels = ['Indemne ', 'Blessé hospitalisé', 'Blessé léger'],
			        explode = [0, 0.2, 0],
			        autopct = lambda x : str(round(x, 2)) + '%',
			        pctdistance = 0.7,
			        labeldistance = 1.1,
			        shadow = True,
			        colors = ("lightgreen", "orange", "lightyellow"),
			        startangle=80);
			st.pyplot(fig)
			st.markdown("Pour les hommes, il y a environ la moitié des d’accidents sans blessure et une autre moitié "
			"avec blessures légères. Pour les femmes, c’est plutôt 1/4 d’accidents sans blessure et 3/4 "
			"avec des blessures légères. Pour les deux, la proportion d'accidents entraînant une "
			"hospitalisation est assez faible, d’environ 2%. Sur la période (4 mois et 644 accidents), il n’y "
			"a eu qu’un seul mort, un homme.")



		#accidentés par âge
		####################
		if select_diag == diag2:

			#graphe age
			fig = plt.figure(figsize=(8, 8))
			plt.title("Cyclistes accidentés selon l'âge", fontsize=15)
			df_age = df_acc.groupby('age', as_index = False).agg({'Num_Acc':'count'})
			bins = pd.IntervalIndex.from_tuples([(0, 12), (12, 18), (18, 30), (30, 40), (40, 50), (60,70), (70,150)])
			df_age["cat_age"] = pd.cut(df_age["age"], bins)
			df_age = df_age.groupby('cat_age', as_index = False).agg({'Num_Acc':'sum'})
			plt.pie(x = df_age.Num_Acc,
			        labels = ['Moins de 12 ans',
			                  '12 - 18 ans',
			                  '18 - 30 ans',
			                  '30 - 40 ans',
			                  '40 - 50 ans',
			                  '60 -70 ans',
			                  'Plus de 70 ans'],
			        explode = [0, 0, 0, 0, 0, 0, 0],
			        autopct = lambda x : str(round(x, 2)) + '%',
			        pctdistance = 0.8,
			        labeldistance = 1.1,
			        shadow = True);
			st.pyplot(fig)
			st.markdown("Sans surprise la majorité des accidentés ont entre 18 et 30 ans, ce qui correspond à la tranche "
			"d’âge qui roule le plus à vélo. Puis le nombre décroît avec l’âge (et l’usage).")


		#accidentés par trajet
		####################
		if select_diag == diag3:

			#graphe trajet
			fig = plt.figure(figsize=(8, 8))
			plt.title("Cyclistes accidentés selon la nature du trajet", fontsize=15)
			df_traj = df_acc.groupby('trajet', as_index = False).agg({'trajet':'count'})
			df_traj.replace(-1, 0)
			plt.pie(x = df_traj.trajet,
			        labels = ['Non renseigné',
			                  'Domicile – travail',
			                  'Domicile – école',
			                  'Courses – achats ',
			                  'Utilisation professionnelle',
			                  'Promenade – loisirs',
			                  'Autre'],
			        explode = [0, 0.09, 0, 0, 0, 0.09, 0],
			        autopct = lambda x : str(round(x, 2)) + '%',
			        pctdistance = 0.7,
			        labeldistance = 1.1,
			        shadow = True,
			        startangle = 90);
			st.pyplot(fig)
			st.markdown("Avec 1/3 des trajets non renseignés, difficile de conclure, même si les trajets "
			"Promenade/loisirs et Domicile-travail semblent largement en tête.")
			

		#accidentés par voie
		####################
		if select_diag == diag4:
			#graphe situation
			fig = plt.figure(figsize=(8, 8))
			plt.title("Cyclistes accidentés selon la voie utilisée", fontsize=15)
			df_situ = df_acc.groupby('situ', as_index = False).agg({'situ':'count'})
			df_situ.replace(-1, 0)
			plt.pie(x = df_situ.situ,
			        labels = ['Sur chaussée',
			                  'Sur trottoir',
			                  'Sur piste cyclable',
			                  'Sur autre voie spéciale'],
			        explode = [0, 0, 0.2, 0],
			        autopct = lambda x : str(round(x, 2)) + '%',
			        pctdistance = 0.7,
			        labeldistance = 1.1,
			        shadow = True,
			        startangle = 90);
			st.pyplot(fig)
			st.markdown("Une grosse moitié des accidents a eu lieu sur la chaussée contre 1/3 sur des pistes cyclables. "
			"Ces dernières limiteraient donc le nombre d’accidents.")


		#graphe météo
		if select_diag == diag5:
			fig = plt.figure(figsize=(8, 8))
			plt.title("Cyclistes accidentés selon la météo", fontsize=12)
			df_atm = df_acc.groupby('atm', as_index = False).agg({'atm':'count'})

			plt.pie(x = df_atm.atm,
			        labels = ['Normale',
			                  'Pluie légère',
			                  'Pluie forte',
			                  'Brouillard',
			                  'Temps éblouissant',
			                  'Temps couvert'],
			        explode = [0, 0.1, 0.05, 0.2, 0, 0.1],
			        autopct = lambda x : str(round(x, 2)) + '%',
			        pctdistance = 0.7,
			        labeldistance = 1.1,
			        shadow = True,
			        radius= 0.8);
			st.pyplot(fig)
			st.markdown("Les 3/4 des accidents ont lieu sous une météo normale. Le 1/4 restant a lieu sous la pluie ou "
			"par temps couvert.")



######################
## MACHINE LEARNING ##
######################
if select_theme == 'Machine Learning':
	st.sidebar.header(select_theme)

	label1="Comptages sur les derniers jour du mois"
	label2="Comptages sur les derniers mois de la période"

	select_pred_ml = st.sidebar.radio(
	    "Sélectionnez la période à prédire :",
	    (label1,
	    label2))
	st.sidebar.markdown("<hr>", unsafe_allow_html=True)



#Prédiction sur les derniers jours de chaque mois
#################################################
	if select_pred_ml == label1:
		st.title("Machine Learning")
		st.header(machine_learning1)

		st.sidebar.subheader('Sélectionnez les paramètres du modèle :')
		st.sidebar.markdown("<hr>", unsafe_allow_html=True)

		df_ml = df_ml.sort_values(by = ['Jour'])

		liste_var = st.sidebar.multiselect('Sélectionnez les variables :',
									['Année', 'Mois', 'Jour',
								    'Jour_de_la_semaine', 'Heure', 'Grève', 'Confinement',
								    'Jour_férié', 'Vacances',
								    'vac_aout', 'vac_noel',
								    'Pluie', 'sam_dim', 'lat', 'long',
								    'Comptage_horaire_h_1', 'Comptage_horaire_h_2',
								    'Comptage_horaire_h_3', 'Comptage_horaire_j_1', 'Comptage_horaire_j_2',
								    'Comptage_horaire_j_3', 'Comptage_horaire_s_1', 'Comptage_horaire_s_2',
								    'Comptage_horaire_s_3', 'Comptage_horaire_s_4'],
							       default=['Comptage_horaire_h_1',
									       'Comptage_horaire_h_2',
									       'Comptage_horaire_h_3',
									       'Comptage_horaire_j_1',
									       'Comptage_horaire_j_2',
									       'Comptage_horaire_j_3',
									       'Comptage_horaire_s_1',
									       'Comptage_horaire_s_2',
									       'Comptage_horaire_s_3',
									       'Comptage_horaire_s_4'])
		st.sidebar.markdown("<hr>", unsafe_allow_html=True)

		jour_deb_test = st.sidebar.slider(label = "Sélectionnez le premier jour de prédictions :",
		    min_value = 20, max_value = 31, step = 1, value = 24)
		st.sidebar.markdown("<hr>", unsafe_allow_html=True)

		##calcul du % à prendre pour la taille de l'échantillon test :
		#nb de lignes df_ml
		len_df_ml = len(df_ml)
		#nb de lignes échantillon test
		len_df_test = len(df_ml[df_ml["Jour"] >= jour_deb_test])
		test_size = len_df_test / len_df_ml
		test_size_round = round(test_size*100, 2)

		st.write("Taille échantillon de test = ", (test_size_round), " %")


		data = df_ml[liste_var]
		target = df_ml['Comptage_horaire']
		X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = test_size, shuffle = False)



		algo = st.sidebar.radio(
		    "Sélectionnez l'algorithme à tester :",
		    ('LinearRegression',
		    'Ridge',
		    'Lasso',
		    'ElasticNet',
		    'DecisionTreeRegressor',
		    'RandomForestRegressor',
		    'BaggingRegressor',
		    'GradientBoostingRegressor'))
		st.sidebar.markdown("<hr>", unsafe_allow_html=True)		

		st.subheader('Affichage du score du modèle choisi :')

		if algo == 'LinearRegression':
			modele = LinearRegression()
		elif algo == 'Ridge':
			modele = RidgeCV(alphas = (0.001, 0.01, 0.1, 0.3, 0.7, 1, 10, 50, 100))
		elif algo == 'Lasso':
			modele = LassoCV(alphas = (0.001, 0.01, 0.1, 0.3, 0.7, 1, 10, 50, 100))
		elif algo == 'ElasticNet':
			modele = ElasticNet()
		elif algo == 'DecisionTreeRegressor':
			modele = DecisionTreeRegressor()
		elif algo == 'RandomForestRegressor':
			modele = RandomForestRegressor(n_estimators = 10, criterion = 'mse')
		elif algo == 'BaggingRegressor':
			modele = BaggingRegressor(n_estimators = 10)
		else :
			modele = GradientBoostingRegressor(n_estimators = 100)

		arrondi = 3

		if st.sidebar.checkbox("Standardiser les données", value=False):
			scaler = preprocessing.StandardScaler().fit(X_train) 
			X_train_scaled = scaler.transform(X_train)
			X_test_scaled = scaler.transform(X_test)

			modele.fit(X_train_scaled, y_train)
			pred_train = modele.predict(X_train_scaled)
			pred_test = modele.predict(X_test_scaled)
			st.write(algo, " :")
			st.write("score R² train = ", round(modele.score(X_train_scaled, y_train),arrondi),
					" / score R² test = ",round(modele.score(X_test_scaled, y_test),arrondi)
					)
			st.write("rmse train = ", round(np.sqrt(mean_squared_error(y_train, pred_train)),arrondi),
				     " / rmse test = ", round(np.sqrt(mean_squared_error(y_test, pred_test)),arrondi)
				     )
		else:
			modele.fit(X_train, y_train)
			pred_train = modele.predict(X_train)
			pred_test = modele.predict(X_test)
			st.write(algo, " :")
			st.write("score R² train = ", round(modele.score(X_train, y_train),arrondi),
					" / score R² test = ",round(modele.score(X_test, y_test),arrondi)
					)
			st.write("rmse train = ", round(np.sqrt(mean_squared_error(y_train, pred_train)),arrondi),
				     " / rmse test = ", round(np.sqrt(mean_squared_error(y_test, pred_test)),arrondi)
				     )
		st.sidebar.markdown("<hr>", unsafe_allow_html=True)			



		liste_sites = sorted(df_ml['Nom du site de comptage'].unique().tolist())

		site = st.sidebar.selectbox('Sélectionnez le site de comptage à prédire :', (liste_sites), index = 3)
		st.sidebar.markdown("<hr>", unsafe_allow_html=True)	


		#Définition des variables pour représenter les prévisions :
		select_mois = st.sidebar.radio(
		    "Sélectionnez le mois à prédire :",
		    ('Octobre 2019',
		    'Novembre 2019',
		    'Décembre 2019',
		    'Janvier 2020',
		    'Février 2020',
		    'Mars 2020',
		    'Avril 2020',
		    'Mai 2020',
		    'Juin 2020',
		    'Juillet 2020',
		    'Août 2020',
		    'Septembre 2020',
		    'Octobre 2020',
		    'Novembre 2020',
		    'Décembre 2020'))



		if select_mois == 'Octobre 2019':
			mois = 10
			annee = 2019
		elif select_mois == 'Novembre 2019':
			mois = 11
			annee = 2019
		elif select_mois == 'Décembre 2019':
			mois = 12
			annee = 2019
		elif select_mois == 'Janvier 2020':
			mois = 1
			annee = 2020
		elif select_mois == 'Février 2020':
			mois = 2
			annee = 2020
		elif select_mois == 'Mars 2020':
			mois = 3
			annee = 2020
		elif select_mois == 'Avril 2020':
			mois = 4
			annee = 2020
		elif select_mois == 'Mai 2020':
			mois = 5
			annee = 2020
		elif select_mois == 'Juin 2020':
			mois = 6
			annee = 2020
		elif select_mois == 'Juillet 2020':
			mois = 7
			annee = 2020
		elif select_mois == 'Août 2020':
			mois = 8
			annee = 2020
		elif select_mois == 'Septembre 2020':
			mois = 9
			annee = 2020
		elif select_mois == 'Octobre 2020':
			mois = 10
			annee = 2020
		elif select_mois == 'Novembre 2020':
			mois = 11
			annee = 2020
		else :
			mois = 12
			annee = 2020



		st.subheader('Représentation graphique des prédictions :')	

		# graphe mensuel détail jours

		df_test = pd.DataFrame({'Comptages_prédits' : pred_test.astype('int32')}, index = X_test.index)
		df_pred = df_ml.join(df_test)

		df_graphe = df_pred[(df_pred["Mois"] == mois) &
		                    (df_pred["Année"] == annee) &
		                    (df_pred["Nom du site de comptage"] == site)]

		df_graphe_j = df_graphe.groupby(['Jour','Identifiant du site de comptage','Nom du site de comptage'],
		                              as_index = False).agg({'Comptage_horaire':'sum', 'Comptages_prédits':'sum'})

		titre = str(site) + "\n" + str(select_mois) + "\n(détail journalier)"

		fig = plt.figure(figsize = (16, 5))
		plt.plot(df_graphe_j['Jour'], df_graphe_j['Comptage_horaire'], label= "comptages observés", color ="blue", lw = 1)
		plt.plot(df_graphe_j['Jour'][df_graphe_j['Jour'] >= jour_deb_test],
		            df_graphe_j['Comptages_prédits'][df_graphe_j['Jour'] >= jour_deb_test],
		            color = "red",
		            label= "comptages prédits",
		            lw = 1)
		plt.xlabel("Jour")
		plt.ylabel("Comptages journaliers")
		plt.xticks(np.arange(0, 32, 1))
		plt.title(titre)
		plt.grid(True, linestyle = ':')
		plt.legend();
		st.pyplot(fig)

		# graphe derniers jours du mois détail heures

		df_graphe_h = df_graphe.groupby(['Date et heure de comptage','Identifiant du site de comptage',
		                             'Jour','Nom du site de comptage'], as_index = False).agg({'Comptage_horaire':'sum', 'Comptages_prédits':'sum'})

		titre = str(site) + "\n" + str(select_mois) + "\n(détail heures derniers jours du mois)"

		fig = plt.figure(figsize = (16, 5))
		plt.plot(df_graphe_h['Date et heure de comptage'][df_graphe_h['Jour'] >= jour_deb_test],
		         df_graphe_h['Comptage_horaire'][df_graphe_h['Jour'] >= jour_deb_test],
		         label= "comptages observés",
		         color ="blue", lw = 1)
		plt.plot(df_graphe_h['Date et heure de comptage'][df_graphe_h['Jour'] >= jour_deb_test],
		            df_graphe_h['Comptages_prédits'][df_graphe_h['Jour'] >= jour_deb_test],
		            color = "red",
		            label= "comptages prédits",
		            lw = 1)
		plt.xlabel("Jours / Heures")
		plt.ylabel("Comptages horaires")
		plt.xticks(np.arange(12, len(df_graphe_h["Jour"][df_graphe_h["Jour"] >= jour_deb_test].unique())*24+12, 24),
					df_graphe_h["Jour"][df_graphe_h["Jour"] >= jour_deb_test].unique())
		plt.title(titre)
		plt.grid(True, linestyle = ':')
		plt.legend();
		st.pyplot(fig)



#Prédiction des comptages pour les derniers mois de la période
################################################################
	if select_pred_ml == label2:
		st.title("Machine Learning")
		st.header(machine_learning2)


		st.sidebar.subheader('Sélectionnez les paramètres du modèle :')
		st.sidebar.markdown("<hr>", unsafe_allow_html=True)	

		df_ml = df_ml.sort_values(by = ['Année', 'Mois', 'Jour'])

		liste_var = st.sidebar.multiselect('Sélectionnez les variables :',
									['Année', 'Mois', 'Jour',
								    'Jour_de_la_semaine', 'Heure', 'Grève', 'Confinement',
								    'Jour_férié', 'Vacances',
								    'vac_aout', 'vac_noel',
								    'Pluie', 'lat', 'long', 'sam_dim',
								    'Comptage_horaire', 'Comptage_horaire_h_1', 'Comptage_horaire_h_2',
								    'Comptage_horaire_h_3', 'Comptage_horaire_j_1', 'Comptage_horaire_j_2',
								    'Comptage_horaire_j_3', 'Comptage_horaire_s_1', 'Comptage_horaire_s_2',
								    'Comptage_horaire_s_3', 'Comptage_horaire_s_4'],
							       default=['Comptage_horaire_h_1',
									       'Comptage_horaire_h_2',
									       'Comptage_horaire_h_3',
									       'Comptage_horaire_j_1',
									       'Comptage_horaire_j_2',
									       'Comptage_horaire_j_3',
									       'Comptage_horaire_s_1',
									       'Comptage_horaire_s_2',
									       'Comptage_horaire_s_3',
									       'Comptage_horaire_s_4'])
		st.sidebar.markdown("<hr>", unsafe_allow_html=True)



		#mois_deb_test = st.slider(label = "choix du premier mois de prédictions :",
		#    min_value = 1, max_value = 12, step = 1, value = 10)

		select_mois_deb_test = st.sidebar.radio("Sélectionnez le premier mois des prédictions :",
									    ('Septembre 2020',
									    'Octobre 2020',
									    'Novembre 2020',
									    'Décembre 2020'), index = 1)
		st.sidebar.markdown("<hr>", unsafe_allow_html=True)


		if select_mois_deb_test == 'Janvier 2020':
			mois_deb_test = 1
		elif select_mois_deb_test == 'Février 2020':
			mois_deb_test = 2
		elif select_mois_deb_test == 'Mars 2020':
			mois_deb_test = 3
		elif select_mois_deb_test == 'Avril 2020':
			mois_deb_test = 4
		elif select_mois_deb_test == 'Mai 2020':
			mois_deb_test = 5
		elif select_mois_deb_test == 'Juin 2020':
			mois_deb_test = 6
		elif select_mois_deb_test == 'Juillet 2020':
			mois_deb_test = 7
		elif select_mois_deb_test == 'Août 2020':
			mois_deb_test = 8
		elif select_mois_deb_test == 'Septembre 2020':
			mois_deb_test = 9
		elif select_mois_deb_test == 'Octobre 2020':
			mois_deb_test = 10
		elif select_mois_deb_test == 'Novembre 2020':
			mois_deb_test = 11
		elif select_mois_deb_test == 'Décembre 2020':
			mois_deb_test = 12

		annee_deb_test = 2020,


		##calcul du % à prendre pour la taille de l'échantillon test :

		#nb de lignes df_ml
		len_df_ml = len(df_ml)
		#nb de lignes échantillon test
		len_df_test2 = len(df_ml[(df_ml["Mois"] >= mois_deb_test) & (df_ml["Année"] >= annee_deb_test)])
		test_size2 = len_df_test2 / len_df_ml
		test_size_round2 = round(test_size2*100, 2)

		if mois_deb_test == 12:
			st.write("Période de prédictions : Décembre 2020")
		else :
			st.write("Période de prédictions : De ", select_mois_deb_test, "à Décembre 2020")

		st.write("Taille échantillon de test = ", (test_size_round2), " %")



		data = df_ml[liste_var]
		target = df_ml['Comptage_horaire']
		X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = test_size2, shuffle = False)



		algo = st.sidebar.radio(
		    "Sélectionnez l'algorithme à tester :",
		    ('LinearRegression',
		    'Ridge',
		    'Lasso',
		    'ElasticNet',
		    'DecisionTreeRegressor',
		    'RandomForestRegressor',
		    'BaggingRegressor',
		    'GradientBoostingRegressor'))
		st.sidebar.markdown("<hr>", unsafe_allow_html=True)

		st.subheader('Score du modèle choisi :')

		if algo == 'LinearRegression':
			modele = LinearRegression()
		elif algo == 'Ridge':
			modele = RidgeCV(alphas = (0.001, 0.01, 0.1, 0.3, 0.7, 1, 10, 50, 100))
		elif algo == 'Lasso':
			modele = LassoCV(alphas = (0.001, 0.01, 0.1, 0.3, 0.7, 1, 10, 50, 100))
		elif algo == 'ElasticNet':
			modele = ElasticNet()
		elif algo == 'DecisionTreeRegressor':
			modele = DecisionTreeRegressor()
		elif algo == 'RandomForestRegressor':
			modele = RandomForestRegressor(n_estimators = 10, criterion = 'mse')
		elif algo == 'BaggingRegressor':
			modele = BaggingRegressor(n_estimators = 10)
		else :
			modele = GradientBoostingRegressor(n_estimators = 100)

		arrondi = 3


		if st.sidebar.checkbox("Standardiser les données", value=False):
			scaler = preprocessing.StandardScaler().fit(X_train) 
			X_train_scaled = scaler.transform(X_train)
			X_test_scaled = scaler.transform(X_test)

			modele.fit(X_train_scaled, y_train)
			pred_train = modele.predict(X_train_scaled)
			pred_test = modele.predict(X_test_scaled)
			st.write(algo, " :")
			st.write("score R² train = ", round(modele.score(X_train_scaled, y_train),arrondi),
					" / score R² test = ",round(modele.score(X_test_scaled, y_test),arrondi)
					)
			st.write("rmse train = ", round(np.sqrt(mean_squared_error(y_train, pred_train)),arrondi),
				     " / rmse test = ", round(np.sqrt(mean_squared_error(y_test, pred_test)),arrondi)
				     )
		else:
			modele.fit(X_train, y_train)
			pred_train = modele.predict(X_train)
			pred_test = modele.predict(X_test)
			st.write(algo, " :")
			st.write("score R² train = ", round(modele.score(X_train, y_train),arrondi),
					" / score R² test = ",round(modele.score(X_test, y_test),arrondi)
					)
			st.write("rmse train = ", round(np.sqrt(mean_squared_error(y_train, pred_train)),arrondi),
				     " / rmse test = ", round(np.sqrt(mean_squared_error(y_test, pred_test)),arrondi)
				     )
		st.sidebar.markdown("<hr>", unsafe_allow_html=True)



		#Définition des variables pour représenter les prévisions :

		liste_sites = sorted(df_ml['Nom du site de comptage'].unique().tolist())

		site = st.sidebar.selectbox('Sélectionnez le site de comptage à prédire', (liste_sites), index = 3)
		st.sidebar.markdown("<hr>", unsafe_allow_html=True)


		if mois_deb_test == 9:
			select_mois = st.sidebar.radio(
		    "Sélectionnez le mois à prédire :",
		    ('Septembre 2020',
		    'Octobre 2020',
		    'Novembre 2020',
		    'Décembre 2020'))
		elif mois_deb_test == 10:
			select_mois = st.sidebar.radio(
		    "Sélectionnez du mois à prédire :",
		    ('Octobre 2020',
		    'Novembre 2020',
		    'Décembre 2020'))
		elif mois_deb_test == 11:
			select_mois = st.sidebar.radio(
		    "Sélectionnez du mois à prédire :",
		    ('Novembre 2020',
		    'Décembre 2020'))
		elif mois_deb_test == 12:
			select_mois = 'Décembre 2020'
			st.write("Mois à prédire : Décembre 2020")




		if select_mois == 'Octobre 2019':
			mois = 10
			annee = 2019
		elif select_mois == 'Novembre 2019':
			mois = 11
			annee = 2019
		elif select_mois == 'Décembre 2019':
			mois = 12
			annee = 2019
		elif select_mois == 'Janvier 2020':
			mois = 1
			annee = 2020
		elif select_mois == 'Février 2020':
			mois = 2
			annee = 2020
		elif select_mois == 'Mars 2020':
			mois = 3
			annee = 2020
		elif select_mois == 'Avril 2020':
			mois = 4
			annee = 2020
		elif select_mois == 'Mai 2020':
			mois = 5
			annee = 2020
		elif select_mois == 'Juin 2020':
			mois = 6
			annee = 2020
		elif select_mois == 'Juillet 2020':
			mois = 7
			annee = 2020
		elif select_mois == 'Août 2020':
			mois = 8
			annee = 2020
		elif select_mois == 'Septembre 2020':
			mois = 9
			annee = 2020
		elif select_mois == 'Octobre 2020':
			mois = 10
			annee = 2020
		elif select_mois == 'Novembre 2020':
			mois = 11
			annee = 2020
		elif select_mois == 'Décembre 2020':
			mois = 12
			annee = 2020


		st.subheader('Représentation graphique des prédictions :')	

		# graphe mensuel prévisions mois
		df_test = pd.DataFrame({'Comptages_prédits' : pred_test.round(0).astype('int32')}, index = X_test.index)
		df_pred = df_ml.join(df_test)

		df_graphe = df_pred[(df_pred["Mois"] == mois) &
		                    (df_pred["Année"] == annee) &
		                    (df_pred["Nom du site de comptage"] == site)]

		df_graphe = df_graphe.groupby(['Jour','Identifiant du site de comptage','Nom du site de comptage'],
		                              as_index = False).agg({'Comptage_horaire':'sum', 'Comptages_prédits':'sum'})

		titre = str(site) + "\n" + str(select_mois)

		fig = plt.figure(figsize = (16, 5))
		plt.plot(df_graphe['Jour'], df_graphe['Comptage_horaire'], label= "comptages observés", color ="blue", lw = 1)
		plt.plot(df_graphe['Jour'],df_graphe['Comptages_prédits'], color = "red", label= "comptages prédits", lw = 1)

		plt.xlabel("Jour")
		plt.ylabel("Comptages")
		plt.xticks(np.arange(0, 32, 1))
		plt.title(titre)
		plt.grid(True, linestyle = ':')
		plt.legend();
		st.pyplot(fig)