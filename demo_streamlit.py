import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import statsmodels.api

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

#entête
image = Image.open('velo.jpg')
st.image(image, use_column_width=True, width=500)
st.markdown(
"<h2 style='text-align: center'>"
"<strong>Le vélo à Paris, data analyse du trafic cycliste</strong>"
"<br><span style='font-size: smaller'>de septembre 2019 à décembre 2020</span>"
"</h2>"
, unsafe_allow_html=True)

df_date = pd.read_csv("df_date.csv", sep = ";")
#Données par date/heure variables catégorielles

df_site = pd.read_csv("df_site.csv", sep = ";")
#Données par site (coordonnées géo, adresse photo etc)

df1_comptages = pd.read_csv("df1_comptages.csv", sep = ";")
df2_comptages = pd.read_csv("df2_comptages.csv", sep = ";")
#Données comptages par site et date/heure

df_comptages = pd.concat([df1_comptages, df2_comptages], ignore_index=True)
df_comptages["Date et heure de comptage"] = pd.to_datetime(df_comptages["Date et heure de comptage"])
df_comptages['Date'] = df_comptages['Date et heure de comptage'].dt.date
df_date["Date et heure de comptage"] = pd.to_datetime(df_date["Date et heure de comptage"])
df_date['Date'] = df_date['Date et heure de comptage'].dt.date

df_acc = pd.read_csv('df_acc.csv', sep=';')
#concaténation des 4 fichiers sources du NB, déjà retraité des NA, filtré sur la période sept 2019 à déc 2019 et accidents vélo

df1_ml = pd.read_csv('df1_ml.csv', sep=';')
df2_ml = pd.read_csv('df2_ml.csv', sep=';')
df3_ml = pd.read_csv('df3_ml.csv', sep=';')
df4_ml = pd.read_csv('df4_ml.csv', sep=';')
df_ml = pd.concat([df1_ml, df2_ml, df3_ml, df4_ml], ignore_index=True)
#concaténation des 4 fichiers sources du ML avec variables numériques préalablement créées

df_2019 = pd.read_csv('df_2019.csv', sep=';')
#fichier sources cartographies, flitrés par sites, doublons url photos déjà traités

#dates = pd.read_csv('dates.csv', sep=';')

######################
###Plan de l'appli####
######################
st.sidebar.subheader("Le vélo à Paris, data analyse du trafic cycliste")
page1 = "Projet & résultats"
page2 = "Jeux de données"
page3 = "Cartographie"
page4 = "Évolution temporelle"
page5 = "Trafic & accidents"
page6 = "Prédiction du trafic"

pages = [page1, page2, page3, page4, page5, page6]
select_page = st.sidebar.radio("", pages)

st.sidebar.info(
"Auteurs : "
"Benoit Gascou "
"[linkedIn](https://www.linkedin.com/in/benoît-gascou-53306218b/), "
"Cynthia Laboureau "
"[linkedIn](https://www.linkedin.com/in/cynthia-lab/), "
"Joséphine Vaton "
"[linkedIn](https://www.linkedin.com/in/josephine-vaton-3a311695/)"
"\n\n"
"En formation de Data Analyst, "
"[DataScientest](https://datascientest.com/), "
"Bootcamp novembre 2020"
"\n\n"
"Données :"
"\n"
"[Ville de Paris](https://opendata.paris.fr/explore/dataset/comptage-velo-donnees-compteurs/information/?disjunctive.id_compteur&disjunctive.nom_compteur&disjunctive.id&disjunctive.name), "
"[data.gouv.fr](https://www.data.gouv.fr/fr/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2019/)"
)


##########################
####Projet & résultats####
##########################
if select_page == page1:
	#Titres, auterus, sources
	#st.markdown("<h2 style='text-align: center; color: black;'>Évolution du trafic cycliste à Paris<br> de septembre 2019 à décembre 2020</h3>", unsafe_allow_html=True)
	st.markdown("<div style='border: 1px solid black; padding: 5px;'>"
		"<p style='text-align: center; color: black;'>Projet réalisé dans le cadre de la formation <strong>Data Analyst</strong> de <a href='https://www.linkedin.com/school/datascientest/'>DataScientest.com</a>"
		"<br>Promotion Bootcamp novembre 2020</p>"
		"<p style='text-align: center; color: black;'>Auteurs :"
		"<br><strong>Benoit Gascou </strong><a href='https://www.linkedin.com/in/benoît-gascou-53306218b/'>LinkedIn</a>"
		"<br><strong>Cynthia Laboureau </strong><a href='https://www.linkedin.com/in/cynthia-lab/'>LinkedIn</a>"
		"<br><strong>Joséphine Vaton </strong><a href='https://www.linkedin.com/in/josephine-vaton-3a311695/'>LinkedIn</a>"
		"</p>"
		"<p style='text-align: center; color: black;'>Sources de données :"
		"<br><a href='https://opendata.paris.fr/explore/dataset/comptage-velo-donnees-compteurs/information/?disjunctive.id_compteur&disjunctive.nom_compteur&disjunctive.id&disjunctive.name'>Comptage vélos | Open Data | Ville de Paris</a>"
		"<br><a href='https://www.data.gouv.fr/fr/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2019/'>Bases de données accidents de la circulation routière | data.gouv.fr</a>"
		"<br></p>"
		"</div>", unsafe_allow_html=True)
	#CONTEXTE
	st.markdown("<h4 style='text-align: left; color: black;'><br>I. CONTEXTE</h4>", unsafe_allow_html=True)
	st.markdown("<br><p style='text-align: justify'>"
	"Lors du premier déconfinement en mai 2020, la Mairie de Paris a créé ex nihilo cinquante kilomètres de pistes cyclables. "
	"Le but ? Désengorger les transports en commun pour éviter la propagation du virus tout en limitant le report sur les voitures particulières. "
	"Une politique pro-vélo amorcée depuis longtemps, mais soudain amplifiée par la pandémie. "
	"Aujourd’hui, il suffit de descendre dans la rue pour le constater : il n’y a jamais eu autant de vélos à Paris. "
	"</p>", unsafe_allow_html=True)
	st.markdown("<p style='text-align: justify'>"
	"Nous avons voulu étudier et quantifier cette évolution du trafic cycliste à partir des données disponibles sur le site de la Mairie de Paris, "
	"pour la période du 1er septembre 2019 au 31 décembre 2020. "
	"</p>", unsafe_allow_html=True)
	st.markdown("<p style='text-align: justify'>"
	"Nous avons anglé nos analyses selon quatre axes :<br>"
	"<strong><span style='color: #1ca2d1'>1. Cartographie</span></strong> : le trafic est-il le même partout ?<br>"
	"<strong><span style='color: #1ca2d1'>2. Évolution temporelle</span></strong> : quels facteurs influencent le trafic ?<br>"
	"<strong><span style='color: #1ca2d1'>3. Trafic & accidents</span></strong> : quelles relations ?<br>"
	"<strong><span style='color: #1ca2d1'>4. Prédiction du trafic</span></strong> : modèles de machine learning"
	"</p>", unsafe_allow_html=True)
	st.markdown("<br><p style='font-style: italic'>"
	"Pour accéder à nos analyses détaillées (Data visualisation, Machine learning), cliquez dans le menu de gauche. "
	"Quant à nos résultats, vous les trouverez ci-dessous. "
	"</p>", unsafe_allow_html=True)
	#RESULTATS
	st.markdown("<h4 style='text-align: left; color: black;'><br>II. RESULTATS</h4>", unsafe_allow_html=True)
	st.markdown("<p style='font-size: large'><br><strong>1. Cartographie :</strong> le trafic est-il le même partout ?</p>", unsafe_allow_html=True)
	st.markdown("<p style='text-align: justify'>"
	"Lors de la période étudiée (septembre 2019 - décembre 2020), "
	"<strong><span style='color: #1ca2d1'>69 sites de comptage</span></strong> ont produit des données, "
	"mais seulement la moitié a été opérationnelle tout au long des 16 mois. "
	"C’est pourquoi nous utilisons comme indicateur du trafic le "
	"<strong><span style='color: #1ca2d1'>Comptage moyen/site/heure</span></strong>."
	"</p>"
	"<p style = 'text-decoration: underline'><strong>Top/flop : une forte disparité géographique</strong></p>"
	"<p style='text-align: justify'>"
	"Le site du <strong>boulevard de Magenta</strong> arrive en tête du classement avec un cumul de "
	"<strong><span style='color: #1ca2d1'>289 vélos/heure en moyenne</span></strong>. "
	"Le trafic y est <strong><span style='color: #1ca2d1'>100 fois plus important</span></strong> "
	"qu’à <strong>Porte d’Orléans</strong>, le site le moins fréquenté, avec <strong>"
	"<span style='color: #1ca2d1'>3 vélos/heure en moyenne.</span></strong>"
	"</p>"
	"<p style = 'text-decoration: underline'><strong>De forts contrastes centre/périphérie et rive droite/rive gauche</strong></p>"
	"<p style='text-align: justify'>"
	"A notre carte dynamique Folium, nous avons ajouté un modèle de <strong>clustering, "
	"l’algorithme des K-Means</strong>, pour classer les sites de comptage selon l’intensité du trafic."
	"</p>"
	"<p style='text-align: justify'>"
	"<strong>Zones à trafic élevé</strong>"
	"</ul>"
	"<li>Hypercentre de Paris : Sébastopol, Châtelet, Odéon</li>"
	"<li>10e : Gare du Nord, Gare de l'Est</li>"
	"<li>11e : Popincourt, Père-Lachaise</li>"
	"<li>17e : Avenue de Clichy</li>"
	"<ul>"
	"</p>"
	"<p style='text-align: justify'>"
	"<strong>Zones à trafic modéré</strong>"
	"</ul>"
	"<li>Les quais de Seine</li>"
	"<li>8e : Champs-Elysées</li>"
	"<li>14e : Denfert-Rochereau, Porte de Vanves</li>"
	"<li>15e : Lecourbe, Javel</li>"
	"<li>19e : La Villette</li>"	
	"<ul>"
	"</p>"
	"<p style='text-align: justify'>"
	"<strong>Zones à trafic faible</strong>"
	"</ul>"
	"<li>Quartiers périphériques : Porte de la Chapelle, Porte de Bagnolet, Porte de Charenton, Porte d’Italie, Porte d'Orléans, Porte Maillot</li>"
	"<li>6e, 7e : Duroc, Boulevard du Montparnasse</li>"
	"<li>13e : Place d’Italie</li>"	
	"<ul>"
	"</p>"
	"<p style='text-align: justify'>"
	"<strong><span style='color: #1ca2d1'>"
	"Le trafic est plus intense dans le centre et rive droite, plus faible dans les quartiers périphériques et rive gauche</span></strong>. "
	"D’un côté des quartiers plus denses en activités économiques, commerciales et lieux de sortie, "
	"avec des distances à parcourir plus courtes et un trafic routier congestionné. "
	"De l’autre des quartiers plus résidentiels et plus éloignés des nœuds d’activité."
	"</p>"
	"<p style='text-align: justify'>"
	"<strong><span style='color: #1ca2d1'>Les axes Nord-Sud, Est-Ouest et les quais de Seine</span></strong> "
	"sont quant à eux des points de passage obligés pour traverser Paris à vélo."
	"</p>"		
	, unsafe_allow_html=True)
	st.markdown("<p style='font-size: large'><br><strong>2. Évolution temporelle :</strong> quels facteurs influencent le trafic cycliste ?</p>", unsafe_allow_html=True)
	st.markdown(
	"<p>Voici les chiffres à retenir sur l’évolution du trafic cycliste en fonction des facteurs périodiques, récurrents et exceptionnels.</p>"
	"<br>"
	"<p style='text-decoration: underline'><strong>Périodicité : hausse annuelle et dissemblance semaine/week-end</strong>"
	"<ul>"
	  "<li>Automne 2020 : <span style='color: #1ca2d1; font-weight: bold'>+21 %</span> (vs automne 2019), avec un pic de <span style='color: #1ca2d1; font-weight: bold'>+66 %</span> en septembre</li>"
	  "<li>Du lundi au vendredi : <span style='color: #1ca2d1; font-weight: bold'>2 pics aux heures de pointe</span> (8-9h et 17-19h) qui disparaissent le week-end</li>"
	  "<li>Week-end : <span style='color: #1ca2d1; font-weight: bold'>-32 %</span> (-25 % le samedi, -40 % le dimanche vs du lundi au vendredi)</li>"
	"</ul></p><br>"
	"<p style='text-decoration: underline'><strong>Facteurs récurrents : baisse les jours chômés et puissant impact de la météo</strong>"
	"<ul>"
	  "<li>Jours fériés : <span style='color: #1ca2d1; font-weight: bold'>-46 %</span></li>"
	  "<li>Vacances : <span style='color: #1ca2d1; font-weight: bold'>-15 %</span> (-43 % à Noël, -33 % en février, + 34% en juillet)</li>"
	  "<li>Météo : <span style='color: #1ca2d1; font-weight: bold'>-33 %</span> par temps pluvieux, "
	  "<span style='color: #1ca2d1; font-weight: bold'>-27 %</span> sous 4 °C, <span style='color: #1ca2d1; font-weight: bold'>+50 %</span> au-delà de 25 °C</li>"
	"</ul></p><br>"
	"<p style='text-decoration: underline'><strong>Facteurs exceptionnels : coup de boost des grèves et Covid-19, baisse mitigée pour les confinements</strong>"
	"<ul>"
	  "<li>Grève des transports : <span style='color: #1ca2d1; font-weight: bold'><span style='color: #1ca2d1; font-weight: bold'><span style='color: #1ca2d1; font-weight: bold'><span style='color: #1ca2d1; font-weight: bold'>+58 %</span></li>"
	  "<li>Pandémie de Covid-19 : <span style='color: #1ca2d1; font-weight: bold'><span style='color: #1ca2d1; font-weight: bold'><span style='color: #1ca2d1; font-weight: bold'>+23 %</span></li>"
	  "<li>Confinement strict : <span style='color: #1ca2d1; font-weight: bold'><span style='color: #1ca2d1; font-weight: bold'>-82 %</span></li>"
	  "<li>Confinement souple : <span style='color: #1ca2d1; font-weight: bold'>-36 %</span></li>"
	"</ul></p>"
	"<p style='text-align: justify'>Le facteur le plus pérenne sur la période étudiée est la pandémie de Covid-19. "
	"D’après le <a href='https://www.insee.fr/fr/statistiques/5012724'>Bilan démographique de l’INSEE</a> paru en janvier 2021, "
	"la part du vélo dans les moyens de transport atteignait début 2020, avant la pandémie, <strong>6 % à Paris</strong> et 3 % en France. "
	"Soit une hausse de <strong>50 % en 5 ans.</strong></p>"
	"<p style='text-align: justify'>Nous démontrons une hausse supplémentaire de <strong>23 %</strong> à Paris depuis le début de la pandémie. "
	"Outre la création et l’aménagement de pistes cyclables par la Mairie de Paris, plusieurs causes peuvent l’expliquer :"
	"<ul>"
	  "<li>évitement des transports publics,</li>"
	  "<li>redécouverte de la proximité,</li>"
	  "<li>exercice physique encore praticable,</li>"
	  "<li>aides publiques pour faire réparer ou acheter un vélo.</li>"
	"</ul>"
	"</p>"
	, unsafe_allow_html=True)
	st.markdown("<p style='font-size: large'><br><strong>3. Trafic & accidents de vélos :</strong> quelles relations ?</p>", unsafe_allow_html=True)	
	st.markdown("<br><p style='text-align: justify'>"
	"Les données 2020 n'étant pas disponibles, notre étude porte sur la période <strong>septembre - décembre 2020 (4 mois) : "
	"644 accidents impliquant des vélos et avec des dommages corporels</strong>. "
	"Les tendances observées seront à confirmer sur un plus grand échantillon."
	"</p>"
	"<p style='text-decoration: underline'><strong>Analyse par heure : heures de pointe moins accidentogènes, nuits plus accidentogènes</strong></p>"
	"<p style='text-align: justify'>"
	"En moyenne, on observe <span style='color: #1ca2d1; font-weight: bold'>0,22 accidents de vélos/heure</span> à Paris. "
	"Dans 90% des cas, il n’y a pas d’accident. Dans <span style='color: #1ca2d1; font-weight: bold'>8 %</span> des cas,"
	"il y a <span style='color: #1ca2d1; font-weight: bold'>1 à 2 accidents</span> de vélos/heure. "
	"</p>"
	"<p style='text-align: justify'>"
	"La p-value du test ANOVA est inférieure à 5%, on rejette donc l'hypothèse selon laquelle le "
	"<strong>comptage moyen/site/heure</strong> n'influe pas sur le <strong>nombre d'accidents à Paris/heure.</strong>"
	"</p>"
	"<p style='text-align: justify'>"
	"<span style='color: #1ca2d1; font-weight: bold'>Les heures de pointes</span> (8-9h et 17-20h), où le trafic cycliste est pourtant le plus intense, "
	"<span style='color: #1ca2d1; font-weight: bold'>semblent moins accidentogènes que les autres heures de la journée</span>. "
	"Les usagers utilisant le vélo quotidiennement pour aller travailler seraient-ils plus aguerris ?"
	"</p>"
	"<p style='text-align: justify'>"
	"<span style='color: #1ca2d1; font-weight: bold'>La différence la plus notable se situe entre le jour et la nuit</span> : les ratios accidents-trafic "
	"les plus élevés se situent tous entre minuit et 6h du matin. "
	"Le manque de visibilité et la conduite en état d'ivresse sont des pistes à étudier."
	"</p>"
	"<p style='text-decoration: underline'><strong>Analyse par jour : pas d’évolution majeure du ratio Accidents-Trafic</strong></p>"
	"<p style='text-align: justify'>"
	"En moyenne, on observe <span style='color: #1ca2d1; font-weight: bold'>5,3 accidents de vélos/jour</span> à Paris. "
	"Dans <span style='color: #1ca2d1; font-weight: bold'>79 %</span> des cas, "
	"il y a de <span style='color: #1ca2d1; font-weight: bold'>0 et 7 accidents</span> de vélos/jour. "
	"Dans 95%, il y a moins de 12 accidents/jour, le maximum étant de 19."
	"</p>"	
	"<p style='text-align: justify'>"
	"Entre le 1er septembre et le 31 décembre 2019, on ne note <strong>pas d'évolution majeure du ratio Accidents-Trafic.</strong> "
	"Nous avons aussi regardé son évolution en fonction du mois, "
	"de la semaine et du jour de la semaine, sans observer de changement notable."
	"</p>"
	"<p style='text-decoration: underline'><strong>Cartographie des accidents : ils augmentent avec la circulation</strong></p>"
	"<p style='text-align: justify'>"
	"<strong> Les accidents sont plus nombreux aux grands carrefours</strong> (Opéra, Saint Lazare, Gare de l'Est, Châtelet) "
	"<strong>et sur les grands axes</strong> (Sébastopol, Convention, Lafayette, Belleville), pourtant équipés de pistes cyclables. "
	"Idem sur le boulevard des Maréchaux, dans le sud de Paris. "
	"L'explication est à chercher du côté des véhicules en circulation, bien plus nombreux à ces endroits."
	"</p>"
	"<p style='text-align: justify'>"
	"Il y a eu des accidents partout, mais <strong>l'est parisien est plus touché, par rapport aux 7e "
	"et 16e arrondissements</strong> par exemple, où il y a eu le moins d'accidents. "
	"Cela reflète la proportion d'usagers du vélo. "
	"A l'est, les quartiers sont plus jeunes et plus animés que dans l’ouest parisien."
	"</p>"
	"<p style='text-decoration: underline'><strong>Statistiques sur les cyclistes accidentés</strong></p>"
	"<p style='text-align: justify'>"
	"<ul>"
	  "<li><strong>Sexe</strong> : <span style='color: #1ca2d1; font-weight: bold'>74 % d’hommes, 26 % de femmes</span>.</li>"
	  "<li><strong>Gravité</strong> : pour les hommes, 1/2 des d’accidents sans blessure et 1/2 avec blessures légères. "
	  "Pour les femmes, 1/4 d’accidents sans blessure et 3/4 avec des blessures légères. "
	  "Pour les deux, <span style='color: #1ca2d1; font-weight: bold'>la proportion d'accidents entraînant une hospitalisation est faible (2 %)</span>. " 
	  "En 4 mois, il n’y a eu qu’<span style='color: #1ca2d1; font-weight: bold'>un mort</span>, un homme.</li>"
	  "<li><strong>Age</strong> : sans surprise la majorité des cyclistes accidentés ont "
	  "<span style='color: #1ca2d1; font-weight: bold'>entre 18 et 30 ans (35 %)</span>. "
	  "C’est en effet la tranche d’âge qui roule le plus à vélo. Le nombre décroît ensuite avec l’âge (et l’usage du vélo).</li>"
	  "<li><strong>Type de trajet</strong> : <span style='color: #1ca2d1; font-weight: bold'>Promenade/loisirs (25 %)</span> et "
	  "<span style='color: #1ca2d1; font-weight: bold'>Domicile-travail (22 %)</span> en tête.</li>"
	  "<li><strong>Type de voie</strong> : <span style='color: #1ca2d1; font-weight: bold'>54 % sur la chaussée, 37 % sur une piste cyclable</span>.</li>"
	  "<li><strong>Météo</strong> : <span style='color: #1ca2d1; font-weight: bold'>75 % par météo normale, 14 % sous la pluie</span>.</li>"		
	"</ul>"
	"</p>"					
	, unsafe_allow_html=True)
	st.markdown("<p style='font-size: large'><br><strong>4. Prédiction du trafic</strong></p>", unsafe_allow_html=True)		
	st.markdown(
	"<p style='text-decoration: underline'><strong>Démarche</strong></p>"
	"<p style='text-align: justify'>"
	"<ul>"
	  "<li><strong>Le but</strong> : prédire la variable cible <span style='color: #1ca2d1; font-weight: bold'>Comptage/heure/site</span>.</li>"
	  "<li><strong>Le problème</strong> : 25 variables à disposition, mais aucune véritable variable numérique. "
	  "Elles sont toutes catégorielles et ont peu de relations linéaires avec la variable cible.</li>"
	  "<li><strong>La solution explorée</strong> : créer 10 variables explicatives numériques, dérivées de <span style='color: #1ca2d1; font-weight: bold'>Comptage/heure/site</span> "
	  "(Comptage à Heure -1, -2, -3, à Jour -1, -2, -3, à Semaine -1, -2, -3, -4).</li>"
	"</ul>"
	"</p>"
	"<p style='text-decoration: underline'><strong>Prédictions sur la dernière semaine de chaque mois</strong></p>"
	"<p style='text-align: justify'>"
	"<ul>"
	  "<li><strong>Choix des variables</strong> (SelectKBest et SelectFrom Model) : nous gardons les 10 variables numériques, mais pas les autres.</li>"
	  "<li><strong>Choix du modèle</strong> : LinearRegression, le plus robuste parmi les 8 modèles de régression testés.</li>"
	  "<li><strong>Entrainement</strong> du 1er au 23 de chaque mois et test à partir du 24</li>"
	  "<li><strong>Performance +++</strong> : "
	  "<span style='color: #1ca2d1; font-weight: bold'>R² train/test : 0.92 / 0.91 - RMSE train/test : 34.2 / 33.7</span></li>"
	  "<li><strong>Représentations graphiques</strong> : </li>"
	"</ul>"
	"</p>"
	, unsafe_allow_html=True)
	graphe_1 = Image.open('graphe_1.png')
	graphe_2 = Image.open('graphe_2.png')	
	st.image(graphe_1, use_column_width=True, width=500)
	st.image(graphe_2, use_column_width=True, width=500)	
	st.markdown(
	"<p style='text-decoration: underline'><strong>Prédictions sur les derniers mois de la période</strong></p>"
	"<p style='text-align: justify'>"
	"<ul>"
	  "<li><strong>Choix des variables</strong> : nous ajoutons 3 variables explicatives <span style='color: #1ca2d1; font-weight: bold'>(vac_noel, jours_fériés, jour_de_la_semaine)</span></li>"
	  "<li><strong>Choix du modèle</strong> : LinearRegression, le plus robuste parmi les 8 modèles de régression testés.</li>"
	  "<li><strong>Entrainement</strong> sur 12 mois (oct 2019 - sep 2020- et <strong>test</strong> sur 4 mois (sept - déc 2020)</li>"
	  "<li><strong>Performance ++</strong> : "
	  "<span style='color: #1ca2d1; font-weight: bold'>R² train/test =  0.92  /  0.91 - RMSE train/test =  34.2  /  31.7</span></li>"
	  "<li><strong>Représentations graphiques</strong> : </li>"
	"</ul>"
	"</p>"
	, unsafe_allow_html=True)
	graphe_3 = Image.open('graphe_3.png')
	st.image(graphe_3, use_column_width=True, width=500)
	st.markdown(
	"<p style='text-decoration: underline'><strong>Conclusions</strong></p>"
	"<p style='text-align: justify'>"
	"<ul>"
	  "<li>Nous avons obtenu un très bon modèle “théorique” de régression linéaire.</li>"
	  "<li><strong>La limite</strong> : en le “nourrissant” de toutes les heures précédant le comptage, les prévisions réelles sont impossibles.</li>"
	  "<li><strong>L’étape suivante</strong> : créer un modèle capable de prévisions à partir de données nouvelles.</li>"
	"</ul>"
	"</p>"
	, unsafe_allow_html=True)
	#PERSPECTIVES
	st.markdown("<h4 style='text-align: left; color: black;'><br>III. PERSPECTIVES</h4>", unsafe_allow_html=True)
	st.markdown("<br><p style='text-align: justify'>"
	"Pour accompagner la <strong>Mairie de Paris</strong> et la <strong>Sécurité Routière</strong> dans leurs décisions en matière d’aménagement urbain, "
	"il nous semble important de poursuivre et approfondir cette étude. "
	"</p>", unsafe_allow_html=True)
	st.markdown("<p style='text-align: justify'>"
	"L’étape suivante consiste à <span style='color: #1ca2d1; font-weight: bold'>monitorer le trafic cycliste à Paris</span>, "
	"en récoltant les données en temps réel. Pour analyser plus finement les causes et les conséquences de la variation du trafic, "
	"il est aussi important d’<span style='color: #1ca2d1; font-weight: bold'>élargir la période étudiée aux données archivées</span>. "
	"Nous pouvons aussi <span style='color: #1ca2d1; font-weight: bold'>pousser plus loin les études d’impact</span> "
	"sur les accidents de vélos et sur d’autres thèmes, comme les retombées sur l’usage de la voiture ou des transports publics."
	"</p>"
	"<p style='text-align: justify'>"
	"Concernant les prédictions, il nous faut à présent "
	"<span style='color: #1ca2d1; font-weight: bold'>passer à une phase de prévisions du trafic</span> à court et moyen termes.</span>"
	"</p>"
	, unsafe_allow_html=True)
	st.markdown("<p style='text-align: justify'>"
	"A cette fin, nous identifions plusieurs pistes d’amélioration :"	
	"<ul>"
	  "<li>"
	  "<strong>Relevé des données “Comptage vélos / heure / site”</strong>"
	    "<ul>"
	      "<li>Renforcer le maillage des sites de comptage</li>"
	      "<li>Équiper les axes Est-Ouest, sous-exploités</li>"
	     "</ul>"
	  "</li>"
	  "<li>"
	  "<strong>Ajout de données extérieures (historiques et actuelles)</strong>"
	    "<ul>"
	      "<li>Accidents impliquant des vélos / heure</li>"
	      "<li>Relevés météorologiques / heure</li>"
	      "<li>Trafic routier / heure</li>"
	      "<li>Affluence dans les transports en commun / heure</li>"
	     "</ul>"
	  "</li>"
	  "<li>"
	  "<strong>Machine et deep learning</strong>"
	    "<ul>"
	      "<li>Créer de nouvelles variables explicatives numériques et indépendantes, grâce aux données extérieures</li>"
	      "<li>Relevés météorologiques / heure</li>"
	      "<li>Tester d’autres types de modèles, capables de prévisions sur les séries temporelles, comme les Microsoft Time Series ou les Réseaux de Neurones.</li>"
	     "</ul>"
	  "</li>"
	"</ul>"
	"</p>", unsafe_allow_html=True)

##########################
####Jeux de données   ####
##########################
elif select_page == page2:
	st.header(select_page)
	dataset_1 = "Dataset Principal"
	dataset_2 = "Dataset Secondaire"
	select_dataset = st.radio("", [dataset_1, dataset_2])
	if select_dataset == dataset_1:
		st.subheader("DATASET PRINCIPAL")
		st.markdown("<h3>1. Source</h3>"
		"<p style='text-align: justify'>"
		"Le jeu de données provient du site de la Mairie de Paris : "
		"<a href='https://opendata.paris.fr/explore/dataset/comptage-velo-donnees-compteurs/information/?disjunctive.id_compteur&disjunctive.nom_compteur&disjunctive.id&disjunctive.name'>Comptage Vélo – Données compteurs</a>"
		"</p>", unsafe_allow_html=True)
		st.markdown("<h3>2. Période</h3>"
		"<p style='text-align: justify'>"
		"Les données sont mises à jour quotidiennement et remontent sur 13 mois glissants. "
		"Nous avons récupéré toutes les données <strong>du 1er septembre 2019 au 31 décembre 2020 (16 mois).</strong>"
		"</p>", unsafe_allow_html=True)
		st.markdown("<h3>3. Remarques</h3>"
		"<p style='text-align: justify'>"
		"<ul>"
		   "<li>Les données sont fournies par un prestataire : <a href='https://www.eco-compteur.com/'>Eco-Compteur</a>.</li>"
		   "<li>Le nombre de compteurs fluctue. Certains sont créés, d’autres arrêtés en cas de travaux ou de panne.</li>"		  
		"</ul>"
		"</p>", unsafe_allow_html=True)
		st.markdown("<h3>4. Exploration des données</h3>", unsafe_allow_html=True)
		st.markdown("<p style='text-align: justify'>"
		"Langage utilisé : <strong>Python</strong><br>"
		"Librairies utilisées : <strong>Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn</strong>"
		"</p>"
		"<p style='text-align: justify'>"
		"<strong>Taille du DataFrame</strong><br>"
		"1 002 827 lignes x 9 colonnes"
		"</p>"
		"<p style='text-align: justify'>"
		"<strong>Les variables</strong><br>"
		"<table style='text-align: center ; margin: auto'>"
		  "<tr>"
		    "<th>N° colonne</th>"
		    "<th>Nom de la colonne</th>"
		    "<th>Type</th>"
		  "</tr>"
		  "<tr>"
		    "<td>1</td>"
		    "<td>Identifiant du compteur</td>"
		    "<td>string</td>"
		  "</tr>"
		  "<tr>"
		    "<td>2</td>"
		    "<td>Nom du compteur</td>"
		    "<td>string</td>"
		  "</tr>"
		  "<tr>"
		    "<td>3</td>"
		    "<td>Identifiant du site de comptage</td>"
		    "<td>string</td>"
		  "</tr>"
		  "<tr>"
		    "<td>4</td>"
		    "<td>Nom du site de comptage</td>"
		    "<td>string</td>"
		  "</tr>"
		  "<tr>"
		    "<td>5</td>"
		    "<td>Comptage horaire</td>"
		    "<td>integer</td>"
		  "</tr>"
		  "<tr>"
		    "<td>6</td>"
		    "<td>Date et heure de comptage</td>"
		    "<td>DateTime</td>"
		  "</tr>"
		  "<tr>"
		    "<td>7</td>"
		    "<td>Date d'installation du site de comptage</td>"
		    "<td>date</td>"
		  "</tr>"	
		  "<tr>"
		    "<td>8</td>"
		    "<td>Lien vers photo du site de comptage</td>"
		    "<td>string</td>"
		  "</tr>"
		  "<tr>"
		    "<td>9</td>"
		    "<td>Coordonnées géographiques</td>"
		    "<td>geo_point_2d</td>"
		  "</tr>"		  		  	  		  
		"</table>"
		"</p>"
		"<p style='text-align: justify'>"
		"La variable cible est le “<span style='color: #1ca2d1'>Comptage horaire</span>”, c’est-à-dire <strong><span style='color: #1ca2d1'>le nombre de vélos/heure/site</span></strong>. "
		"C’est la seule variable numérique continue."
		"</p>"
		"<p style='text-align: justify'>"
		"Nous n’avons que deux variables explicatives, respectivement temporelle et géographique : "
		" “<span style='color: #1ca2d1'>Date et heure du comptage</span>” et “<span style='color: #1ca2d1'>Coordonnées géographiques”</span>."
		"</p>"
		, unsafe_allow_html=True)
		st.markdown("<h3>5. Traitement des données</h3>"
		"<p style='text-align: justify'>"
		"Le DataFrame ne contient pas de doublons."
		"</p>"
		"<p style='text-align: justify'>"
		"<strong>Valeurs manquantes</strong><br>"
		"37 152 valeurs sont manquantes, soit 3,7 % du DataFrame. "
		"Elles concernent 6 variables, toutes liées à l'identification des compteurs. "
		"Nous retrouvons les informations via les adresses des sites et nous remplaçons toutes les valeurs manquantes."
		"</p>"
		"<p style='text-align: justify'>"
		"<strong>Valeurs extrêmes de la variable “Comptage horaire”</strong><br>"
		"Les statistiques descriptives font apparaître des valeurs extrêmes (3e quartile : 68, 4e quartile : 1275). "
		"Un boxplot nous permet de les visualiser : elles sont nombreuses, continues et cohérentes. "
		"Un minimum de 0 vélo/heure/site et un maximum de 1 275 (soit 21 vélos/minute/site) ne sont pas aberrants. "
		"Nous gardons toutes les valeurs."
		"</p>"
		, unsafe_allow_html=True)
		st.markdown("<h3>6. Ajout de variables</h3>"
		"<p style='text-align: justify'>"
		"Pour alimenter notre analyse, nous créons 25 variables."
		"</p>"
		"<p style='text-align: justify'>"
		"<strong>Périodicité</strong> : Année, Mois, Semaine, Jour de la semaine, Week-end etc."
		"</p>"
		"<p style='text-align: justify'>"
		"<strong>Événements récurrents</strong> : Jours fériés, Types de vacances, Infos météo (Pluie, Froid, Chaleur)."
		"</p>"
		"<p style='text-align: justify'>"
		"<strong>Evènements exceptionnels</strong> : Grève, Covid, Confinement."
		"</p>"		
		, unsafe_allow_html=True)
		st.markdown("<br><strong> Extrait du DataFrame</strong>", unsafe_allow_html=True)
		st.dataframe(df_ml.iloc[:,0:19].head(10))
	elif select_dataset == dataset_2:
		st.subheader("DATASET SECONDAIRE")
		st.markdown("<h3>1. Source</h3>"
		"<p style='text-align: justify'>"
		"Ce deuxième jeu de données provient de la Plateforme ouverte des données publiques françaises : "
		"<a href='https://www.data.gouv.fr/fr/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2019/'>"
		"Bases de données annuelles des accidents corporels de la circulation routière - Années de 2005 à 2019</a>"
		"</p>", unsafe_allow_html=True)
		st.markdown("<h3>2. Période</h3>"
		"<p style='text-align: justify'>"
		"Nous sélectionnons la période <strong>du 1er septembre au 31 décembre 2019 (4 mois)</strong>, les données 2020 n’étant pas encore disponibles."
		"</p>", unsafe_allow_html=True)
		st.markdown("<h3>3. Remarques</h3>"
		"<p style='text-align: justify'>"
		"Une étude sur 4 mois nous permet déjà d'identifier les liens entre l’intensité du trafic cycliste et le nombre d’accidents impliquant des vélos. "
		"D’autant que sur la période septembre - décembre 2019, la grève des transports a provoqué une hausse du trafic. "
		"Cela servira de tendance pour l'année 2020, où la pandémie a elle aussi provoqué une hausse."
		"</p>"
		"<p style='text-align: justify'>"
		"Les accidents de cette base sont ceux ayant provoqué des dommages corporels."
		"</p>"
		, unsafe_allow_html=True)
		st.markdown("<h3>4. Traitement des données</h3>"
		"<p style='text-align: justify'>"
		"<ul>"
		   "<li>La base de données comprend plusieurs tables. Nous en réunissons quatre (par la variable identifiant les accidents) : Caractéristiques, Véhicules, Lieux, Usagers</li>"
		   "<li>Nous sélectionnons les colonnes utiles à notre analyse.</li>"
		   "<li>Nous filtrons par ville (Paris), période (sep-déc 2019) et catégorie de véhicules impliqués (vélo).</li>"
		   "<li>Il n’y a ni valeur manquante ni doublon.</li>"
		   "<li>Nous convertissons le type de certaines colonnes.</li>"
		   "<li>Nous créons une colonne ‘Age’ (2019 - année de naissance des victimes).</li>"		  
		"</ul>"
		"</p>", unsafe_allow_html=True)
		st.markdown("<br><strong> Extrait du DataFrame</strong>", unsafe_allow_html=True)
		st.dataframe(df_acc.head(10))
############################
## Évolution géographique ##
############################
elif select_page == page3:
	st.header(select_page)
	carte_1 = "Cartographie générale"
	carte_2 = "Cartographie sur mesure"
	select_carte = st.radio("", [carte_1, carte_2])
	if select_carte == carte_1:
		#Carte générale
		###############
		st.markdown("<p style='text-align: justify'>"
		"Pour la cartographie, nous avons choisi la bibliothèque <strong>Folium</strong>. "
		"Elle permet de géolocaliser des informations sur une carte dynamique et détaillée de type OpenStreetMap."
		"</p>", unsafe_allow_html=True)			
		st.markdown(			
		"<p style='text-align: justify'>"
		"Nous avons utilisé un modèle de <strong>classification non-supervisée (Clustering) :</strong> "
		"<strong>l’algorithme des k-moyennes (K-Means)</strong>. Il affiche les sites de comptage en 3 groupes "
		"(bleu, vert ou rouge) selon de l’intensité du trafic."
		"</p>"
		"<p>La taille des cercles est proportionnelle au <strong>nombre moyen de vélos/heure/site.</strong></p>"
		, unsafe_allow_html=True)
		df_filtre1 = df_date
		# création d'un nouveau df qui servira à la cartographie
		df_geo1 = df_comptages[df_comptages["Date et heure de comptage"].isin(df_filtre1["Date et heure de comptage"])]
		df_geo1 = df_geo1.groupby(['Identifiant du site de comptage'], as_index = False).agg({'Comptage horaire':'mean'})
		df_geo1 = df_geo1.merge(right=df_site, on="Identifiant du site de comptage", how="left")
		# Clustering avec le modèle K-Means
		#Création d'un DF avec les coordonnées géographiques et le comptage horaire moyen pour chaque site de comptage
		df_cluster1 = df_geo1[['Identifiant du site de comptage','Comptage horaire', 'lat', 'long']]
		kmeans1 = KMeans(n_clusters = 3)
		kmeans1.fit(df_cluster1[["Comptage horaire", "lat", "long"]])
		centroids1 = kmeans1.cluster_centers_
		labels1 = kmeans1.labels_
		# Ajouter les labels au DF "df_cluster"
		labels1 = pd.DataFrame(labels1)
		df_cluster1 = df_cluster1.join(labels1)
		df_cluster1["groupe"] = df_cluster1[0]
		df_cluster1 = df_cluster1.drop(0, axis = 1)
		# Fusionner le DF "df_cluster" avec le précédent DF servira à la cartographie
		df_geo1 = df_geo1.join(df_cluster1["groupe"])
		# affichage des sites de comptage sur la carte en utilisant la librairie Folium
		carte1 = folium.Map(location = [48.86, 2.341886], zoom_start = 12, min_zoom=12)

		for nom1, comptage1, latitude1, longitude1, image1, groupe1 in zip(df_geo1["Nom du site de comptage"],
		                                                             df_geo1["Comptage horaire"],
		                                                             df_geo1["lat"],
		                                                             df_geo1["long"],
		                                                             df_geo1["Lien vers photo du site de comptage"],
		                                                             df_geo1["groupe"]):
			if groupe1 == 0:
				couleur1 = "#d9152a"
			elif groupe1 == 1:
				couleur1 = "#368fe5"
			else:
				couleur1 = "#129012"
			if comptage1 == 0:
				rayon1 = 0.5
			else:
				rayon1 = comptage1
			pp1 = "<strong>" + nom1 + "</strong>" + "<br>nombre moyen de vélos/heure/site : " + str(round(comptage1,2)) + "<br><img src='" + image1 + "', width=100%>"
			folium.CircleMarker(location=[latitude1, longitude1],
			                    radius=rayon1/10,
			                    popup = folium.Popup(pp1, max_width = 300),
			                    tooltip= "<strong>" + nom1 + "</strong>",
			                    color=couleur1,
			                    fill_color=couleur1,
			                    fill_opacity=0.3
			                   ).add_to(carte1)
		folium_static(carte1)
		st.markdown(			
		"<p style='font-style: italic ; text-align: justify'>"
		"En cliquant sur les cercles, vous pouvez faire apparaître l’adresse, l’identifiant, "
		"la photo et le nombre moyen de vélos / heure pour chaque site de comptage. "
		"</p>", unsafe_allow_html=True)
		st.subheader("Remarques préliminaires")
		st.markdown(			
		"<p style='text-align: justify'>"
		"Lors de la période étudiée (septembre 2019 - décembre 2020), 69 sites de comptage ont produit des données. "
		"Cependant, certains n’ont pas été présents tout au long des 16 mois. "
		"Il y a eu des créations de site, mais aussi des travaux ou des pannes empêchant d’autres sites de fonctionner."
		"</p>"
		"<p style='text-align: justify'>"
		"Chaque site de comptage comporte un ou deux compteurs. Un si la circulation est à sens unique. Deux si elle est à double sens. "
		"Mais dans certains cas, notamment quand la piste cyclable se situe de part et d'autre d'un boulevard, "
		"deux sites de comptage ont été créés au lieu de deux compteurs pour un seul site. "
		"</p>"
		"<p style='text-align: justify'>"
		"Notre analyse porte sur les 69 sites de comptages (à sens unique ou à double sens) et non sur chaque compteur."
		"</p>"
		, unsafe_allow_html=True)
		st.subheader("Répartition des sites de comptage")
		st.markdown(			
		"<p style='text-align: justify'>"
		"<strong>Point forts</strong>"
		"<ul>"
			  "<li>La couverture générale : les sites sont disséminés un peu partout dans Paris</li>"
			  "<li>Des axes majeurs bien couverts :"
			    "<ul>"
			      "<li>Les quais de Seine</li>"
			      "<li>L’axe Nord-Sud : Gare du Nord/Gare de l'Est - Châtelet - Odéon - Alésia</li>"
			     "</ul>"
			  "</li>"
		"</ul></p>"
		"<p style='text-align: justify'>"
		"<strong>Point faibles</strong>"
		"<ul>"
			  "<li>Faible maillage : 3,5 sites/arrondissement</li>"
			  "<li>Des arrondissements peu ou pas couverts : 1er, 2e, 9e, 16e, 18e, 20e"			  
			  "<li>Des axes Est-Ouest, pourtant très fréquentés, non couverts :"
			    "<ul>"
			      "<li>Porte de Vincennes - Nation - Bastille - Saint-Paul</li>"
			      "<li>Porte de Bagnolet - Père-Lachaise - République - Saint-Lazare</li>"
			     "</ul>"
			  "</li>"
		"</ul>""</p>"
		, unsafe_allow_html=True)
		st.subheader("Analyse géographique du trafic (nombre moyen de vélos/heure/site)")
		st.markdown(			
		"<p style='text-align: justify'>"
		"Les trois catégories de sites de comptage permettent une première analyse géographique du trafic cycliste à Paris lors de la période étudiée."
		"</p>"
		"<p style='text-align: justify'>"
		"<strong>Zones à trafic élevé</strong>"
		"<ul>"
			  "<li>Hypercentre de Paris : Sébastopol, Châtelet, Odéon</li>"
			  "<li>10e : Gare du Nord, Gare de l'Est</li>"
			  "<li>11e : Popincourt, Père-Lachaise</li>"
			  "<li>17e : Avenue de Clichy</li>"
		"</ul></p>"
		"<p style='text-align: justify'>"
		"<br><strong>Zones à trafic modéré</strong><br>"
		"<ul>"
			  "<li>Le long des quais de Seine</li>"
			  "<li>8e : Champs-Elysées</li>"
			  "<li>14e : Denfert-Rochereau, Porte de Vanves"
			  "<li>15e : Lecourbe, Javel</li>"
			  "<li>19e : La Villette</li>"			  
		"</ul></p>"
		"<p style='text-align: justify'>"		
		"<br><strong>Zones à trafic faible</strong><br>"
		"<ul>"
			  "<li>Quartiers périphériques : Porte de la Chapelle, Porte de Bagnolet, Porte de Charenton, Porte d’Italie, Porte d'Orléans, Porte Maillot</li>"
			  "<li>6e, 7e : Duroc, Boulevard du Montparnasse</li>"
			  "<li>13e : Place d’Italie</li>"		  
		"</ul></p>"
		"</p>"
		"<p style='text-align: justify'>"
		"On note des contrastes centre/périphérie et rive droite/rive gauche. "
		"Le trafic est globalement plus dense dans le centre et rive droite. "
		"Ce sont en effet des quartiers plus animés, plus denses en activités économiques, commerciales et lieux de sortie. "
		"La distance à parcourir pour les activités quotidiennes est plus courte et le trafic routier souvent congestionné, ce qui favorise l’usage du vélo. "
		"Les axes Nord-Sud, Est-Ouest ou le long des quais sont quant à eux des points de passage obligés pour traverser Paris à vélo. "
		"Les quartiers périphériques et la rive gauche sont plus résidentiels et le trafic cycliste semble en pâtir."
		"</p>"
		, unsafe_allow_html=True)
		st.subheader("Top/flop du trafic")
		df_top_flop = df_comptages.groupby(['Identifiant du site de comptage'], as_index = False).agg({'Comptage horaire':'mean'})
		df_top_flop = df_top_flop.merge(right = df_site, on = "Identifiant du site de comptage")
		df_top = df_top_flop[['Nom du site de comptage', 'Comptage horaire']].sort_values(by='Comptage horaire', ascending = True).tail(10).reset_index()
		df_flop = df_top_flop[['Nom du site de comptage', 'Comptage horaire']].sort_values(by='Comptage horaire', ascending = True).head(10).reset_index()
		fig = plt.figure(figsize = (4, 6))
		plt.subplot(2, 1, 1)
		plt.barh(df_top['Nom du site de comptage'], df_top['Comptage horaire'], color="#1ca2d1")
		plt.xlim(0, 180)
		plt.ylabel('Site de comptage', fontsize = 6)
		plt.xlabel('Nombre moyen de vélos / heure / site', fontsize = 6)
		plt.xticks(fontsize = 6)
		plt.yticks(fontsize = 6)
		plt.title("Top 10", fontsize = 9)
		plt.text(25, 9-0.15, df_top['Comptage horaire'][9].round(1), fontsize=7, color="white", weight="bold")
		plt.text(25, 8-0.15, df_top['Comptage horaire'][8].round(1), fontsize=7, color="white", weight="bold")
		plt.text(25, 7-0.15, df_top['Comptage horaire'][7].round(1), fontsize=7, color="white", weight="bold")
		plt.text(25, 6-0.15, df_top['Comptage horaire'][6].round(1), fontsize=7, color="white", weight="bold")
		plt.text(25, 5-0.15, df_top['Comptage horaire'][5].round(1), fontsize=7, color="white", weight="bold")
		plt.text(25, 4-0.15, df_top['Comptage horaire'][4].round(1), fontsize=7, color="white", weight="bold")
		plt.text(25, 3-0.15, df_top['Comptage horaire'][3].round(1), fontsize=7, color="white", weight="bold")
		plt.text(25, 2-0.15, df_top['Comptage horaire'][2].round(1), fontsize=7, color="white", weight="bold")
		plt.text(25, 1-0.15, df_top['Comptage horaire'][1].round(1), fontsize=7, color="white", weight="bold")
		plt.text(25, 0-0.15, df_top['Comptage horaire'][0].round(1), fontsize=7, color="white", weight="bold")
		plt.gcf().subplots_adjust(hspace = 0.4)
		plt.subplot(2, 1, 2)
		plt.barh(df_flop['Nom du site de comptage'], df_flop['Comptage horaire'], color="#d52b52")
		plt.xlim(0, 180)
		plt.ylabel('Site de comptage', fontsize = 6)
		plt.xlabel('Nombre moyen de vélos / heure / site', fontsize = 6)
		plt.xticks(fontsize = 6)
		plt.yticks(fontsize = 6)		
		plt.title("Flop 10", fontsize = 9)
		plt.text(25, 9-0.15, df_flop['Comptage horaire'][9].round(1), fontsize=7, color="#d52b52", weight="bold")
		plt.text(25, 8-0.15, df_flop['Comptage horaire'][8].round(1), fontsize=7, color="#d52b52", weight="bold")
		plt.text(25, 7-0.15, df_flop['Comptage horaire'][7].round(1), fontsize=7, color="#d52b52", weight="bold")
		plt.text(25, 6-0.15, df_flop['Comptage horaire'][6].round(1), fontsize=7, color="#d52b52", weight="bold")
		plt.text(25, 5-0.15, df_flop['Comptage horaire'][5].round(1), fontsize=7, color="#d52b52", weight="bold")
		plt.text(25, 4-0.15, df_flop['Comptage horaire'][4].round(1), fontsize=7, color="#d52b52", weight="bold")
		plt.text(25, 3-0.15, df_flop['Comptage horaire'][3].round(1), fontsize=7, color="#d52b52", weight="bold")
		plt.text(25, 2-0.15, df_flop['Comptage horaire'][2].round(1), fontsize=7, color="#d52b52", weight="bold")
		plt.text(25, 1-0.15, df_flop['Comptage horaire'][1].round(1), fontsize=7, color="#d52b52", weight="bold")
		plt.text(25, 0-0.15, df_flop['Comptage horaire'][0].round(1), fontsize=7, color="#d52b52", weight="bold")
		st.pyplot(fig)
		st.markdown(			
		"<p style='text-align: justify'>"
		"Le site du <strong>boulevard de Sébastopol</strong> (à double sens) semble le plus fréquenté avec <strong><span style='color: #1ca2d1'>171 vélos/heure</span></strong> en moyenne. "
		"Mais les deux sites suivants correspondent aux deux voies à sens unique de part et d’autre du <strong>boulevard de Magenta</strong>. "
		"</p>"
		"<p style='text-align: justify'>"
		"Celui-ci arrive donc en tête du classement avec un cumul de <strong><span style='color: #1ca2d1'>287 vélos/heure</span></strong> en moyenne "
		"L’écart est énorme avec le site de comptage le moins fréquenté - <strong><span style='color: #1ca2d1'>3 vélos/heure</span></strong> en moyenne à <strong>Porte d’Orléans</strong>."
		"</p>"
		"<p style='text-align: justify'>"
		"Dans le reste du classement, on retrouve sans surprise les sites les plus et moins fréquentés cités précédemment."
		"</p>"
		, unsafe_allow_html=True)		


	elif select_carte == carte_2:
		#Choix de la période
		####################
		st.subheader("A votre tour !")
		st.markdown(			
		"<p style='font-style: italic ; text-align: justify'>"
		"A l’aide des filtres ci-dessous, vous pouvez modifier tous les paramètres et ainsi afficher la carte qui vous intéresse, "
		"par exemple : le trafic de jour ou de nuit, en semaine ou le week-end, pendant les vacances scolaires ou les jours fériés, "
		"lors d’une grève ou d’un confinement…"
		"</p>", unsafe_allow_html=True)	

		st.header("Choix de la période étudiée")
		#st.write("Sélectionnez une période entre le 01/09/2019 et le 31/12/2020")
		col1, col2 = st.beta_columns([2.5, 2])
		with col1:
			date = st.date_input(
				'Sélectionnez une période entre le 01/09/2019 et le 31/12/2020',
				value=(dt.date(2019, 9, 1), dt.date(2020, 12, 31)),
				min_value=dt.date(2019, 9, 1),
				max_value=dt.date(2020, 12, 31))

		st.header("Choix de la périodicité")
		col1, col2 = st.beta_columns([1, 3])
		with col1:
			annee = st.multiselect("Sélectionnez l'année", [2019, 2020], default = [2019, 2020])
		with col2:
			liste_mois = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre"]
			select_mois = st.multiselect("Sélectionnez le mois", liste_mois, default = liste_mois)
			mois =[]
			for i in np.arange(0,12):
				if liste_mois[i] in select_mois :
					mois.append(i+1)	
		col1, col2 = st.beta_columns([3, 2])
		with col1:
			liste_jr_sem = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
			select_jr_sem = st.multiselect('Sélectionnez le jour de la semaine', liste_jr_sem, default = liste_jr_sem)
			jr_sem =[]
			for i in np.arange(0,7):
				if liste_jr_sem[i] in select_jr_sem :
					jr_sem.append(i)
		with col2:		
			heure = st.slider("Sélectionnez la tranche horaire", min_value=0, max_value=23, step=1, value=(0, 23))
			heure2 = ()
			tranche_hor2 = st.checkbox("Ajoutez une tranche horaire", value=False)
			if tranche_hor2:
				heure2 = st.slider("", min_value=0, max_value=23, step=1, value=(0, 23))

		plus_filtres = st.checkbox("Plus de filtres", value = False)
		if plus_filtres:
			liste_semaine = np.arange(1,54).tolist()
			semaine = st.multiselect('Sélectionnez le jour de la semaine (1 à 53)', liste_semaine, default=liste_semaine)
			liste_jour = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
			17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
			jour = st.multiselect('Sélectionnez le jour du mois (1 à 31)', liste_jour, default=liste_jour)

		#Evénements récurrents
		######################
		st.header("Autres choix")

		col1, col2, col3 = st.beta_columns(3)
		#Weekend
		with col1:
			st.subheader("Week-ends")
			weekend = st.checkbox("Du lundi au vendredi", value=True)
			pas_weekend = st.checkbox("Week-ends", value=True)
		#Férié
		with col2:
			st.subheader("Jours fériés")
			ferie = st.checkbox("Hors jours fériés", value=True)
			pas_ferie = st.checkbox("Jours fériés", value=True)

		#Vacances
		st.subheader("Vacances")
		col1, col2, col3 = st.beta_columns(3)
		with col1:
			hors_vac = st.checkbox("Hors vacances scolaires", value=True)
			fevrier = st.checkbox("Vacances de février", value=True)
			printemps = st.checkbox("Vacances de Pâques", value=True)
		with col2:		 
			ascension = st.checkbox("Pont de l'Ascension", value=True)
			ete = st.checkbox("Vacances d'été", value=True)
		with col3:	
			toussaint = st.checkbox("Vacances de la Toussaint", value=True)
			noel = st.checkbox("Vacances de Noël ", value=True)

		#Météo
		st.subheader("Météo")
		col1, col2, col3 = st.beta_columns([1, 1, 1])
		with col1:
			st.markdown("<strong>Pluie</strong>", unsafe_allow_html=True)
			pas_de_pluie = st.checkbox("Hors jours de pluie (précipitations ≤ 10 mm)", value=True)
			pluie = st.checkbox("Jours de pluie (précipitations > 10 mm)", value=True)
		with col2:	
			st.markdown("<strong>Froid</strong>", unsafe_allow_html=True)
			inf_4 = st.checkbox("Hors jours de grand froid (t > 4 °C)", value=True)
			sup_4 = st.checkbox("Jours de grand froid (t ≤ 4 °C)", value=True)
		with col3:	
			st.markdown("<strong>Chaleur</strong>", unsafe_allow_html=True)
			sup_25 = st.checkbox("Hors jours de grosse chaleur (t  < 25 °C)", value=True)
			inf_25 = st.checkbox("Jours de grosse chaleur (t ≥ 25 °C)", value=True)

		#Evénements exceptionnels
		##########################
		col1, col2, col3 = st.beta_columns(3)
		with col1:
			st.subheader("Pandémie de Covid-19")
			av_cov = st.checkbox("Avant la pandémie (< 17/03/2020)", value=True)
			covid = st.checkbox("Depuis la pandémie (≥ 17/03/2020)", value=True)
		with col2:
			st.subheader("Confinement")
			pas_conf = st.checkbox("Pas de confinement", value=True)
			conf_1 = st.checkbox("1er confinement", value=True)
			conf_2 = st.checkbox("2e confinement", value=True)
		with col3:
			st.subheader("Grève des tansports")
			greve = st.checkbox("Hors grève", value=True)
			pas_greve = st.checkbox("Jours de grève", value=True)	

		st.header("Avec ou sans clustering")
		st.markdown("Modèle de machine learning qui classe les sites selon l'intensité du trafic")
		clustering = st.checkbox("Clustering", value=False)

		#Filtres
		########
		df_filtre = df_date
		df_filtre = df_filtre[(df_filtre["Date"] >= date[0]) & (df_filtre["Date"] <= date[1])]
		df_filtre = df_filtre[df_filtre["Année"].isin(annee)]
		df_filtre = df_filtre[df_filtre["Mois"].isin(mois)]
		df_filtre = df_filtre[df_filtre["Jour_de_la_semaine"].isin(jr_sem)]
		if tranche_hor2:
			df_filtre = df_filtre[(df_filtre["Heure"] >= heure[0]) & (df_filtre["Heure"] <= heure[1]) | (df_filtre["Heure"] >= heure2[0]) & (df_filtre["Heure"] <= heure2[1])]
		else:
			df_filtre = df_filtre[(df_filtre["Heure"] >= heure[0]) & (df_filtre["Heure"] <= heure[1])]
		if plus_filtres:
			df_filtre = df_filtre[df_filtre["Jour"].isin(jour)]
			df_filtre = df_filtre[df_filtre["Semaine"].isin(semaine)]
		if weekend == False:
			df_filtre = df_filtre.drop(df_filtre.loc[df_filtre["Weekend"] == 1].index, axis=0)
		if pas_weekend == False:
			df_filtre = df_filtre.drop(df_filtre.loc[df_filtre["Weekend"] == 0].index, axis=0)
		if ferie == False:
			df_filtre = df_filtre.drop(df_filtre.loc[df_filtre["Jour_férié"] == 1].index, axis=0)
		if pas_ferie == False:
			df_filtre = df_filtre.drop(df_filtre.loc[df_filtre["Jour_férié"] == 0].index, axis=0)
		if hors_vac == False:
			df_filtre = df_filtre.drop(df_filtre.loc[df_filtre["Vacances"] == 0].index, axis=0)
		if fevrier == False:
			df_filtre = df_filtre.drop(df_filtre.loc[df_filtre["vac_fevrier"] == 1].index, axis=0)
		if printemps == False:
			df_filtre = df_filtre.drop(df_filtre.loc[df_filtre["vac_printemps"] == 1].index, axis=0)
		if ascension == False:
			df_filtre = df_filtre.drop(df_filtre.loc[df_filtre["vac_ascension"] == 1].index, axis=0)
		if ete == False:
			df_filtre = df_filtre.drop(df_filtre.loc[df_filtre["vac_juillet"] == 1].index, axis=0)
			df_filtre = df_filtre.drop(df_filtre.loc[df_filtre["vac_aout"] == 1].index, axis=0)
		if toussaint == False:
			df_filtre = df_filtre.drop(df_filtre.loc[df_filtre["vac_toussaint"] == 1].index, axis=0)
		if noel == False:
			df_filtre = df_filtre.drop(df_filtre.loc[df_filtre["vac_noel"] == 1].index, axis=0)
		if pas_de_pluie == False:
			df_filtre = df_filtre.drop(df_filtre.loc[df_filtre["Pluie"] == 0].index, axis=0)
		if pluie == False:
			df_filtre = df_filtre.drop(df_filtre.loc[df_filtre["Pluie"] == 1].index, axis=0)
			df_filtre = df_filtre.drop(df_filtre.loc[df_filtre["Pluie"] == 2].index, axis=0)
		if sup_4 == False:
			df_filtre = df_filtre.drop(df_filtre.loc[df_filtre["Froid"] == 0].index, axis=0)
		if inf_4 == False:
			df_filtre = df_filtre.drop(df_filtre.loc[df_filtre["Froid"] == 1].index, axis=0)
		if sup_25 == False:
			df_filtre = df_filtre.drop(df_filtre.loc[df_filtre["Chaud"] == 1].index, axis=0)
		if inf_25 == False:
			df_filtre = df_filtre.drop(df_filtre.loc[df_filtre["Chaud"] == 0].index, axis=0)
		if av_cov == False:
			df_filtre = df_filtre.drop(df_filtre.loc[df_filtre["Covid"] == 0].index, axis=0)
		if covid == False:
			df_filtre = df_filtre.drop(df_filtre.loc[df_filtre["Covid"] == 1].index, axis=0)
		if pas_conf == False:
			df_filtre = df_filtre.drop(df_filtre.loc[df_filtre["Confinement"] == 0].index, axis=0)
		if conf_1 == False:
			df_filtre = df_filtre.drop(df_filtre.loc[df_filtre["Confinement"] == 1].index, axis=0)
		if conf_2 == False:
			df_filtre = df_filtre.drop(df_filtre.loc[df_filtre["Confinement"] == 2].index, axis=0)
		if greve == False:
			df_filtre = df_filtre.drop(df_filtre.loc[df_filtre["Grève"] == 1].index, axis=0)
		if pas_greve == False:
			df_filtre = df_filtre.drop(df_filtre.loc[df_filtre["Grève"] == 0].index, axis=0)	

		# création d'un nouveau df qui servira à la cartographie
		df_geo = df_comptages[df_comptages["Date et heure de comptage"].isin(df_filtre["Date et heure de comptage"])]
		df_geo = df_geo.groupby(['Identifiant du site de comptage'], as_index = False).agg({'Comptage horaire':'mean'})
		df_geo = df_geo.merge(right=df_site, on="Identifiant du site de comptage", how="left")

		# Clustering avec le modèle K-Means
		if clustering:
			from sklearn.cluster import KMeans
			#Création d'un DF avec les coordonnées géographiques et le comptage horaire moyen pour chaque site de comptage
			df_cluster = df_geo[['Identifiant du site de comptage','Comptage horaire', 'lat', 'long']]
			kmeans = KMeans(n_clusters = 3)
			kmeans.fit(df_cluster[["Comptage horaire", "lat", "long"]])
			centroids = kmeans.cluster_centers_
			labels = kmeans.labels_
			# Ajouter les labels au DF "df_cluster"
			labels = pd.DataFrame(labels)
			df_cluster = df_cluster.join(labels)
			df_cluster["groupe"] = df_cluster[0]
			df_cluster = df_cluster.drop(0, axis = 1)
			# Fusionner le DF "df_cluster" avec le précédent DF servira à la cartographie
			df_geo = df_geo.join(df_cluster["groupe"])


			# affichage des sites de comptage sur la carte en utilisant la librairie Folium
			carte = folium.Map(location = [48.86, 2.341886], zoom_start = 12, min_zoom=12)

			for nom, comptage, latitude, longitude, image, groupe in zip(df_geo["Nom du site de comptage"],
			                                                             df_geo["Comptage horaire"],
			                                                             df_geo["lat"],
			                                                             df_geo["long"],
			                                                             df_geo["Lien vers photo du site de comptage"],
			                                                             df_geo["groupe"]):
				if groupe == 0:
					couleur = "#d9152a"
				elif groupe == 1:
					couleur = "#368fe5"
				else:
					couleur = "#129012"
				if comptage == 0:
					rayon = 0.5
				else:
					rayon = comptage
				pp = "<strong>" + nom + "</strong>" + "<br>nombre moyen de vélos/heure/site : " + str(round(comptage,2)) + "<br><img src='" + image + "', width=100%>"
				folium.CircleMarker(location=[latitude, longitude],
				                    radius=rayon/10,
				                    popup = folium.Popup(pp, max_width = 300),
				                    tooltip= "<strong>" + nom + "</strong>",
				                    color=couleur,
				                    fill_color=couleur,
				                    fill_opacity=0.3
				                   ).add_to(carte)
			folium_static(carte)
		else :
			# affichage des sites de comptage sur la carte en utilisant la librairie Folium
			carte = folium.Map(location = [48.86, 2.341886], zoom_start = 12, min_zoom=12)

			for nom, comptage, latitude, longitude, image in zip(df_geo["Nom du site de comptage"],
			                                                             df_geo["Comptage horaire"],
			                                                             df_geo["lat"],
			                                                             df_geo["long"],
			                                                             df_geo["Lien vers photo du site de comptage"]):
				if comptage == 0:
					rayon = 0.5
				else:
					rayon = comptage
				couleur = "#368fe5"
				pp = "<strong>" + nom + "</strong>" + "<br>nombre moyen de vélos/heure/site : " + str(round(comptage,2)) + "<br><img src='" + image + "', width=100%>"
				folium.CircleMarker(location=[latitude, longitude],
				                    radius=rayon/10,
				                    popup = folium.Popup(pp, max_width = 300),
				                    tooltip= "<strong>" + nom + "</strong>",
				                    color=couleur,
				                    fill_color=couleur,
				                    fill_opacity=0.3
				                   ).add_to(carte)
			folium_static(carte)

############################
####Évolution temporelle####
############################
elif select_page == page4:
	st.header(select_page)
	temp1 = "Courbe générale"
	temp2 = "Périodicité"
	temp3 = "Evènements récurrents"
	temp4 = "Evènements exceptionnels"
	dataviz_temp = st.radio("", (temp1, temp2, temp3, temp4))
	#Courbe générale
	################
	if dataviz_temp == temp1:
		st.subheader(dataviz_temp)
		dates = df_comptages.groupby('Date')['Comptage horaire'].mean()
		fig = plt.figure(figsize = (30, 10))
		plt.plot_date(dates.index, dates, 'b-', label = "Nombre moyen de vélos / heure / site")
		plt.xlabel('\nDate', fontsize = 20)
		plt.ylabel('Nombre moyen de vélos / heure / site', fontsize = 20)
		plt.ylim(0, 130)
		plt.title('Trafic cycliste à Paris entre septembre 2019 et décembre 2020\n', fontsize = 22)
		liste_dates = [dt.datetime(2019, 9, 1),
		               dt.datetime(2019, 11, 1),
		               dt.datetime(2020, 1, 1),
		               dt.datetime(2020, 3, 1),
		               dt.datetime(2020, 5, 1),
		               dt.datetime(2020, 7, 1),
		               dt.datetime(2020, 9, 1),
		               dt.datetime(2020, 11, 1),
		               dt.datetime(2021, 1, 1)]
		plt.xticks(liste_dates, ['Sep 2019', 'Nov 2019', 'Jan 2020', 'Mars 2020', 'Mai 2020', 'Juil 2020', 'Sep 2020', 'Nov 2020', 'Jan 2021'],fontsize = 18)
		plt.yticks(fontsize = 18)
		plt.legend(fontsize = 18)
		plt.annotate('Grève des transports', xy=(dt.datetime(2019, 9, 15), 103), xytext=(dt.datetime(2019, 11, 1), 154), fontsize = 22, ha = "left", c = "blue", arrowprops={'facecolor':'black', 'arrowstyle':'->'})
		plt.annotate('', xy=(dt.datetime(2019, 12, 15), 115), xytext=(dt.datetime(2019, 12, 14), 153), fontsize = 22, c = "blue", arrowprops={'facecolor':'black', 'arrowstyle':'->'})
		plt.annotate('', xy=(dt.datetime(2020, 1, 10), 106), xytext=(dt.datetime(2019, 12, 14), 153), fontsize = 22, c = "blue", arrowprops={'facecolor':'black', 'arrowstyle':'->'})
		plt.annotate('Déconfinement', xy=(dt.datetime(2020, 5, 10), 60), xytext=(dt.datetime(2020, 3, 20), 154), c = "blue", fontsize = 22, arrowprops={'facecolor':'black', 'arrowstyle':'->'})
		plt.annotate('Mois de septembre le plus chaud jamais enregistré', xy=(dt.datetime(2020, 9, 15), 115), xytext=(dt.datetime(2020, 9, 5), 154), c = "blue", fontsize = 22 ,ha = "center", arrowprops={'facecolor':'black', 'arrowstyle':'->'})
		plt.annotate('Noël', xy=(dt.datetime(2019, 12, 25), 15), xytext=(dt.datetime(2019, 11, 30), -30), c = "blue", fontsize = 20, arrowprops={'facecolor':'black', 'arrowstyle':'->'})
		plt.annotate("1er confinement", xy=(dt.datetime(2020, 3, 19), 4), xytext=(dt.datetime(2020, 2, 7), -30), c = "blue", fontsize = 22, arrowprops={'facecolor':'black', 'arrowstyle':'->'})
		plt.annotate('Août', xy=(dt.datetime(2020, 8, 10), 25), xytext=(dt.datetime(2020, 7, 15), -30), c = "blue", fontsize = 20, arrowprops={'facecolor':'black', 'arrowstyle':'->'})
		plt.annotate('2e confinement', xy=(dt.datetime(2020, 11, 1), 15), xytext=(dt.datetime(2020, 9, 20), -30), c = "blue", fontsize = 22, arrowprops={'facecolor':'black', 'arrowstyle':'->'})
		plt.annotate('Noël', xy=(dt.datetime(2020, 12, 25), 5), xytext=(dt.datetime(2020, 12, 7), -30), c = "blue", fontsize = 22, arrowprops={'facecolor':'black', 'arrowstyle':'->'})
		st.pyplot(fig)
		st.markdown(
		"<p style='text-align: justify'>"
		"On observe en premier lieu une périodicité hebdomadaire et une fluctuation importante au cours des 16 mois de la période étudiée. "
		"Suivons cette évolution dans le détail."
		"</p>"
		"<p style='text-align: justify'>"
		"Au fil de l’automne le trafic baisse progressivement, le froid et la pluie rebutant les usagers."
		"</p>"
		"<p style='text-align: justify'>"
		"La grande grève des transports débute le 5 décembre et immédiatement le trafic monte en flèche. "
		"Seules les vacances de Noël cassent, provisoirement, la courbe."
		"</p>"
		"<p style='text-align: justify'>"
		"Lorsque les transports publics reprennent le 27 janvier 2020 le trafic diminue, mais reste légèrement supérieur au mois précédent la grève. "
		"Il remonte ensuite à l’approche du printemps."
		"</p>"
		"<p style='text-align: justify'>"
		"Mais lors du 1er confinement, du 17 mars au 10 mai, le trafic s'effondre. "
		"Avec le déconfinement et les beaux jours, il reprend de plus belle. "
		"On observe un creux au mois d'août, dû aux vacances et à l'absence de touristes cette année."
		"</p>"
		"<p style='text-align: justify'>"
		"A la rentrée, le trafic cycliste à Paris atteint des sommets, plus élevés qu’en période de grève. "
		"Il faut dire que le mois de septembre 2020 a été le plus chaud jamais enregistré en France. "
		"Et la crise sanitaire est passée par là. Peur des transports publics, création de 50 km de pistes cyclables par la Mairie de Paris, "
		"sans compter les aides publiques pour l'achat ou la réparation de vélos."
		"Bon nombre de parisiens ont adopté la petite reine comme moyen de transport quotidien."	
		"</p>"
		"<p style='text-align: justify'>"
		"En octobre, le trafic diminue, mais reste largement supérieur à l’automne 2019. "
		"Le 30, c’est le deuxième confinement, moins sévère que le premier. Le trafic diminue en conséquence. "
		"Puis viennent les vacances de Noël, sous couvre-feu, et le trafic s’effondre à nouveau."
		"</p>"
		"<p style='text-align: justify'>"
		"Cette courbe générale nous donne plusieurs axes d’analyse : "
		"la périodicité (mensuelle, hebdomadaire, journalière, horaire), "
		"les évènements récurrents (jour férié, vacances, météo) et les évènements exceptionnels (grève, Covid, confinements)."
		"</p>"
		, unsafe_allow_html=True)

	#Périodicité
	######################
	elif dataviz_temp == temp2:
		st.subheader(dataviz_temp)
		periodi1 = "Mensuelle"
		periodi2 = "Journalière"
		periodi3 = "Horaire"
		select_periodi = st.radio("", (periodi1, periodi2, periodi3))
		if select_periodi == periodi1:
			#Mensuelle
			df_comptages["Mois"] = df_comptages['Date et heure de comptage'].dt.month
			df_comptages["Année"] = df_comptages['Date et heure de comptage'].dt.year
			df_mois = df_comptages.groupby(['Mois', 'Année'], as_index = False)['Comptage horaire'].mean()
			fig = plt.figure(figsize = (5, 3))
			sns.barplot(x = 'Mois', y = 'Comptage horaire' , hue = 'Année', data = df_mois)
			plt.xticks(range(0, 12), ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Dec'])
			plt.xlabel("Mois", fontsize = 7)
			plt.ylabel('Nombre moyen de vélos / heure / site', fontsize = 7)
			plt.title('Trafic cycliste par mois', fontsize = 9)
			plt.xticks(fontsize = 7)
			plt.yticks(np.arange(0, 100, 10), fontsize = 7)
			plt.legend(fontsize = 7)
			st.pyplot(fig)
			st.markdown(
			"<p style='text-align: justify'>"
			"La périodicité mensuelle est fortement impactée par les évènements exceptionnels "
			"décrits dans la courbe générale : grève, Covid et confinements. "
			"</p>"
			"<p style='text-align: justify'>"
			"La hausse du trafic est flagrante entre les automnes 2020 et 2019 : +21 %. Dans le détail : "
			"+66 % en septembre, +31 % en octobre et +26 % en novembre. On observe cependant -39 % "
			"en décembre 2020. En effet, le deuxième confinement et le couvre-feu ont fait chuter le trafic, "
			"alors qu’il était boosté par la grève en décembre 2019."
			"</p>", unsafe_allow_html=True)				
		elif select_periodi == periodi2:
			# Journalière
			df_jr0 = df_comptages[df_comptages["Date et heure de comptage"].isin(df_date[df_date["Jour_de_la_semaine"] == 0]["Date et heure de comptage"])]
			df_jr0["Jour_de_la_semaine"] = 0
			df_jr1 = df_comptages[df_comptages["Date et heure de comptage"].isin(df_date[df_date["Jour_de_la_semaine"] == 1]["Date et heure de comptage"])]
			df_jr1["Jour_de_la_semaine"] = 1
			df_jr2 = df_comptages[df_comptages["Date et heure de comptage"].isin(df_date[df_date["Jour_de_la_semaine"] == 2]["Date et heure de comptage"])]
			df_jr2["Jour_de_la_semaine"] = 2
			df_jr3 = df_comptages[df_comptages["Date et heure de comptage"].isin(df_date[df_date["Jour_de_la_semaine"] == 3]["Date et heure de comptage"])]
			df_jr3["Jour_de_la_semaine"] = 3
			df_jr4 = df_comptages[df_comptages["Date et heure de comptage"].isin(df_date[df_date["Jour_de_la_semaine"] == 4]["Date et heure de comptage"])]
			df_jr4["Jour_de_la_semaine"] = 4
			df_jr5 = df_comptages[df_comptages["Date et heure de comptage"].isin(df_date[df_date["Jour_de_la_semaine"] == 5]["Date et heure de comptage"])]
			df_jr5["Jour_de_la_semaine"] = 5
			df_jr6 = df_comptages[df_comptages["Date et heure de comptage"].isin(df_date[df_date["Jour_de_la_semaine"] == 6]["Date et heure de comptage"])]
			df_jr6["Jour_de_la_semaine"] = 6
			df_graphe = pd.concat([df_jr0, df_jr1, df_jr2, df_jr3, df_jr4, df_jr5, df_jr6], ignore_index=True)
			df_graphe = df_graphe.groupby('Jour_de_la_semaine', as_index = False).agg({'Comptage horaire':'mean'})
			fig = plt.figure(figsize = (5, 3))
			sns.barplot(x=df_graphe.index, y=df_graphe['Comptage horaire'],palette = 'hls')
			plt.title('Trafic selon les jours de la semaine\n', fontsize = 9)
			plt.ylabel('Nombre moyen de vélos moyen / heure', fontsize = 7)
			plt.xticks(range(7), ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'], fontsize = 7)
			plt.yticks(fontsize = 7)
			plt.text(-0.25, 30, df_graphe['Comptage horaire'][0].round(1), fontsize=8, color="white", weight="bold")
			plt.text(0.75, 30, df_graphe['Comptage horaire'][1].round(1), fontsize=8, color="white", weight="bold")
			plt.text(1.75, 30, df_graphe['Comptage horaire'][2].round(1), fontsize=8, color="white", weight="bold")
			plt.text(2.75, 30, df_graphe['Comptage horaire'][3].round(1), fontsize=8, color="white", weight="bold")
			plt.text(3.75, 30, df_graphe['Comptage horaire'][4].round(1), fontsize=8, color="white", weight="bold")
			plt.text(4.75, 30, df_graphe['Comptage horaire'][5].round(1), fontsize=8, color="white", weight="bold")
			plt.text(5.75, 30, df_graphe['Comptage horaire'][6].round(1), fontsize=8, color="white", weight="bold")
			st.pyplot(fig)			
			st.markdown(
			"<p style='text-align: justify'>"
			"Nous voyons bien la périodicité hebdomadaire. "
			"Le trafic cycliste est plus important en semaine que le week-end, avec un pic en milieu de semaine. "
			"Le vélo à Paris n’est donc pas un simple loisir, il est principalement un moyen de transport quotidien."
			"</p>", unsafe_allow_html=True)
			st.markdown(
			"<p style='text-align: justify'>"
			"<br>Quantifions les différences entre le trafic en semaine et le week-end. "
			"</p>", unsafe_allow_html=True)
			#Horaire
			col1, col2 = st.beta_columns(2)
			with col1 :
				df_weekend1 = df_comptages[df_comptages["Date et heure de comptage"].isin(df_date[df_date["Weekend"] == 1]["Date et heure de comptage"])]
				df_weekend1["Weekend"] = 1
				df_weekend0 = df_comptages[df_comptages["Date et heure de comptage"].isin(df_date[df_date["Weekend"] == 0]["Date et heure de comptage"])]
				df_weekend0["Weekend"] = 0
				df_graphe = pd.concat([df_weekend1, df_weekend0], ignore_index=True)
				df_graphe = df_graphe.groupby('Weekend', as_index = False).agg({'Comptage horaire':'mean'})
				fig = plt.figure(figsize = (6, 6))
				sns.barplot(x=df_graphe.index, y=df_graphe['Comptage horaire'], palette = 'hls')
				plt.title('Trafic cycliste semaine vs week-end\n', fontsize = 14)
				plt.ylabel('Nombre moyen de vélos moyen / heure', fontsize = 12)
				plt.xticks(range(2), ['Lundi au Vendredi', 'Weekend'], fontsize = 13)
				plt.text(-0.12, 30, df_graphe['Comptage horaire'][0].round(1), fontsize=12, color="white", weight="bold")
				plt.text(0.89, 30, df_graphe['Comptage horaire'][1].round(1), fontsize=12, color="white", weight="bold")
				st.pyplot(fig)
			with col2 :				
				#Weekend vs samedi & dimanche
				df_week0 = df_comptages[df_comptages["Date et heure de comptage"].isin(df_date[df_date["Weekend"] == 0]["Date et heure de comptage"])]
				df_week0["Week"] = 0
				df_week1 = df_comptages[df_comptages["Date et heure de comptage"].isin(df_date[df_date["Jour_de_la_semaine"] == 5]["Date et heure de comptage"])]
				df_week1["Week"] = 1
				df_week2 = df_comptages[df_comptages["Date et heure de comptage"].isin(df_date[df_date["Jour_de_la_semaine"] == 6]["Date et heure de comptage"])]
				df_week2["Week"] = 2
				df_graphe = pd.concat([df_week0, df_week1, df_week2], ignore_index=True)
				df_graphe = df_graphe.groupby('Week', as_index = False).agg({'Comptage horaire':'mean'})
				fig = plt.figure(figsize = (6, 6))
				sns.barplot(x=df_graphe.index, y=df_graphe['Comptage horaire'], palette = 'hls')
				plt.title('Trafic cycliste semaine vs samedi et dimanche\n', fontsize = 14)
				plt.ylabel('Nombre moyen de vélos moyen / heure', fontsize = 12)
				plt.text(-0.1, 30, df_graphe['Comptage horaire'][0].round(1), fontsize=12, color="white", weight="bold")
				plt.text(0.9, 30, df_graphe['Comptage horaire'][1].round(1), fontsize=12, color="white", weight="bold")
				plt.text(1.9, 30, df_graphe['Comptage horaire'][2].round(1), fontsize=12, color="white", weight="bold")
				plt.xticks(range(3), ['Lundi au Vendredi', 'Samedi', 'Dimanche'], fontsize = 13)
				st.pyplot(fig)
			st.markdown(
			"<p style='text-align: justify'>"
			"Le week-end, on note en moyenne -32 % de vélos par rapport à la semaine : "
			"-25 % le samedi et -40 % le dimanche."
			"</p>", unsafe_allow_html=True)
		elif select_periodi == periodi3:
			# jours de la semaine et heure
			liste_jr_sem = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
			select_jr_sem = st.multiselect('Sélectionnez le(s) jour(s) de la semaine :', liste_jr_sem,
				                           default = ["lundi", "mardi", "mercredi", "jeudi", "vendredi"])
			jr_sem =[]
			for i in np.arange(0,7):
			    if liste_jr_sem[i] in select_jr_sem :
			        jr_sem.append(i)
			df_filtre_j = df_date
			df_filtre_j = df_filtre_j[df_filtre_j["Jour_de_la_semaine"].isin(jr_sem)]
			df_jour = df_comptages[df_comptages["Date et heure de comptage"].isin(df_filtre_j["Date et heure de comptage"])]
			df_jour["Heure"] = df_jour['Date et heure de comptage'].dt.hour
			df_jour = df_jour.groupby(['Heure'], as_index = False).agg({'Comptage horaire':'mean'})
			fig = plt.figure(figsize = (8, 4))
			sns.barplot(x = 'Heure', y = 'Comptage horaire', data = df_jour, errwidth=0)
			plt.ylabel('Nombre moyen de vélos / heure / site')
			plt.ylim(0, 170)
			plt.xticks([0, 4, 8, 12, 16, 20, 24], ['minuit', '4h', '8h', 'midi', '16h', '20h', 'minuit'])
			plt.xlabel(" ")
			plt.title("Trafic selon l'heure et le jour de la semaine\n")
			st.pyplot(fig)
			if select_jr_sem == ["lundi", "mardi", "mercredi", "jeudi", "vendredi"]:
				st.markdown(			
				"<p style='text-align: justify'>"
				"En semaine, on observe 2 pics aux heures de pointe : entre 8 et 9h, puis entre 17 et 19h, "
				"comme pour les autres moyens de transport quotidiens (voiture, transports en commun)."
				"</p>", unsafe_allow_html=True)			
			select_jr_sem2 = st.multiselect('Sélectionnez le(s) jour(s) de la semaine :', liste_jr_sem,
			                                default = ["samedi", "dimanche"])
			jr_sem =[]
			for i in np.arange(0,7):
			    if liste_jr_sem[i] in select_jr_sem2 :
			        jr_sem.append(i)
			df_filtre_j = df_date
			df_filtre_j = df_filtre_j[df_filtre_j["Jour_de_la_semaine"].isin(jr_sem)]
			df_jour = df_comptages[df_comptages["Date et heure de comptage"].isin(df_filtre_j["Date et heure de comptage"])]
			df_jour["Heure"] = df_jour['Date et heure de comptage'].dt.hour
			df_jour = df_jour.groupby(['Heure'], as_index = False).agg({'Comptage horaire':'mean'})
			fig = plt.figure(figsize = (8, 4))
			#titres = ['Lundi', 'Mardi','Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
			sns.barplot(x = 'Heure', y = 'Comptage horaire', data = df_jour, errwidth=0)
			plt.ylabel('Nombre moyen de vélos / heure / site')
			plt.ylim(0, 170)
			plt.xticks([0, 4, 8, 12, 16, 20, 24], ['minuit', '4h', '8h', 'midi', '16h', '20h', 'minuit'])
			plt.xlabel(" ")
			plt.title("Trafic selon l'heure et le jour de la semaine\n")
			st.pyplot(fig)
			if select_jr_sem2 == ["samedi", "dimanche"]:			
				st.markdown(			
				"<p style='text-align: justify'>"
				"Le week-end, pas de pics. La courbe est plus lisse avec une bosse en fin d'après-midi "
				"(16-17h) et un creux au milieu de la nuit (4-5h)."
				"</p>", unsafe_allow_html=True)

	#Evènements récurrents
	######################
	elif dataviz_temp == temp3:
		st.subheader(dataviz_temp)
		recur1 = "Jours fériés"
		recur2 = "Vacances"
		recur3 = "Météo"
		select_recur = st.radio("", (recur1, recur2, recur3))
		if select_recur == recur1:
			#st.markdown(select_recur)
			#Jour férié
			col1, col2, col3 = st.beta_columns([1,2,1])
			with col2:
				df_ferie1 = df_comptages[df_comptages["Date et heure de comptage"].isin(df_date[df_date["Jour_férié"] == 1]["Date et heure de comptage"])]
				df_ferie1["Jour_férié"] = 1
				df_ferie0 = df_comptages[df_comptages["Date et heure de comptage"].isin(df_date[df_date["Jour_férié"] == 0]["Date et heure de comptage"])]
				df_ferie0["Jour_férié"] = 0
				df_graphe = pd.concat([df_ferie1, df_ferie0], ignore_index=True)
				df_graphe = df_graphe.groupby('Jour_férié', as_index = False).agg({'Comptage horaire':'mean'})
				fig = plt.figure(figsize = (6, 6))
				sns.barplot(x=df_graphe.index, y=df_graphe['Comptage horaire'],palette = 'hls')
				plt.title('Trafic hors jours feriés vs jours fériés\n', fontsize = 15)
				plt.ylabel('Nombre moyen de vélos / heure / site', fontsize = 13)
				plt.xticks(range(2), ['Hors jours fériés', 'Jours fériés'], fontsize = 13)
				plt.text(-0.12, 25, df_graphe['Comptage horaire'][0].round(1), fontsize=14, color="white", weight="bold")
				plt.text(0.89, 15, df_graphe['Comptage horaire'][1].round(1), fontsize=14, color="white", weight="bold")
				st.pyplot(fig)
			st.markdown(			
			"<p style='text-align: justify'>"
			"Sur la période étudiée (septembre 2019 - décembre 2020), "
			"les jours fériés font baisser le trafic de 46 % en moyenne."
			"</p>", unsafe_allow_html=True)			
		elif select_recur == recur2:
			#Vacances
			col1, col2, col3 = st.beta_columns([1,2,1])
			with col2:
				df_vacances1 = df_comptages[df_comptages["Date et heure de comptage"].isin(df_date[(df_date["Année"] == 2020) & (df_date["Vacances"] == 1)]["Date et heure de comptage"])]
				df_vacances1["Vacances"] = 1
				df_vacances0 = df_comptages[df_comptages["Date et heure de comptage"].isin(df_date[(df_date["Année"] == 2020) & (df_date["Vacances"] == 0)]["Date et heure de comptage"])]
				df_vacances0["Vacances"] = 0
				df_graphe = pd.concat([df_vacances1, df_vacances0], ignore_index=True)
				df_graphe = df_graphe.groupby('Vacances', as_index = False).agg({'Comptage horaire':'mean'})
				fig = plt.figure(figsize = (6, 6))
				sns.barplot(x=df_graphe.index, y=df_graphe['Comptage horaire'],palette = 'hls')
				plt.title('Trafic selon les vacances\n', fontsize = 17)
				plt.ylabel('Nombre moyen de vélos / heure / site', fontsize = 12)
				plt.xticks(range(2), ['Hors vacances', 'Vacances'], fontsize = 12)
				plt.text(-0.12, 30, df_graphe['Comptage horaire'][0].round(1), fontsize=15, color="white", weight="bold")
				plt.text(0.89, 30, df_graphe['Comptage horaire'][1].round(1), fontsize=15, color="white", weight="bold")
				st.pyplot(fig)
			st.markdown(			
			"<p style='text-align: justify'>"
			"Sur la période étudiée (septembre 2019 - décembre 2020), "
			"les vacances scolaires parisiennes font baisser le trafic de 15 % en moyenne."
			"</p><br>", unsafe_allow_html=True)					
			df_vac1 = df_comptages[df_comptages["Date et heure de comptage"].isin(df_date[(df_date["Année"] == 2020) & (df_date["vac_fevrier"] == 1)]["Date et heure de comptage"])]
			df_vac1["vac"] = 0
			df_vac2 = df_comptages[df_comptages["Date et heure de comptage"].isin(df_date[(df_date["Année"] == 2020) & (df_date["vac_printemps"] == 1)]["Date et heure de comptage"])]
			df_vac2["vac"] = 1
			df_vac3 = df_comptages[df_comptages["Date et heure de comptage"].isin(df_date[(df_date["Année"] == 2020) & (df_date["vac_ascension"] == 1)]["Date et heure de comptage"])]
			df_vac3["vac"] = 2
			df_vac4 = df_comptages[df_comptages["Date et heure de comptage"].isin(df_date[(df_date["Année"] == 2020) & (df_date["vac_juillet"] == 1)]["Date et heure de comptage"])]
			df_vac4["vac"] = 3
			df_vac5 = df_comptages[df_comptages["Date et heure de comptage"].isin(df_date[(df_date["Année"] == 2020) & (df_date["vac_aout"] == 1)]["Date et heure de comptage"])]
			df_vac5["vac"] = 4
			df_vac6 = df_comptages[df_comptages["Date et heure de comptage"].isin(df_date[(df_date["Année"] == 2020) & (df_date["vac_toussaint"] == 1)]["Date et heure de comptage"])]
			df_vac6["vac"] = 5
			df_vac7 = df_comptages[df_comptages["Date et heure de comptage"].isin(df_date[(df_date["Année"] == 2020) & (df_date["vac_noel"] == 1)]["Date et heure de comptage"])]
			df_vac7["vac"] = 6
			df_vac0 = df_comptages[df_comptages["Date et heure de comptage"].isin(df_date[(df_date["Année"] == 2020) & (df_date["Vacances"] == 0)]["Date et heure de comptage"])]
			df_vac0["vac"] = 7
			df_graphe = pd.concat([df_vac0, df_vac1, df_vac2, df_vac3, df_vac4, df_vac5, df_vac6, df_vac7], ignore_index=True)
			df_graphe = df_graphe.groupby('vac', as_index = False).agg({'Comptage horaire':'mean'})
			df_graphe = df_graphe.reset_index()
			fig = plt.figure(figsize = (8, 5))
			sns.barplot(x=df_graphe['Comptage horaire'], y=df_graphe.index, orient = 'h',palette = 'hls')
			plt.title("Trafic Vacances vs reste de l'année\n", fontsize = 13)
			plt.xlabel('Nombre moyen de vélospar heure / site', fontsize = 11)
			plt.yticks(range(8), ['Février','Printemps', 'Ascension', 'Juillet', 'Août', 'Toussaint', 'Noël','Hors vacances'], fontsize = 12)
			plt.text(df_graphe['Comptage horaire'][0]-7, 0.1, df_graphe['Comptage horaire'][0].round(1), fontsize=12, color="white", weight="bold")
			plt.text(df_graphe['Comptage horaire'][1]-7, 1+0.1, df_graphe['Comptage horaire'][1].round(1), fontsize=12, color="white", weight="bold")
			plt.text(df_graphe['Comptage horaire'][2]-7, 2+0.1, df_graphe['Comptage horaire'][2].round(1), fontsize=12, color="white", weight="bold")
			plt.text(df_graphe['Comptage horaire'][3]-7, 3+0.1, df_graphe['Comptage horaire'][3].round(1), fontsize=12, color="white", weight="bold")
			plt.text(df_graphe['Comptage horaire'][4]-7, 4+0.1, df_graphe['Comptage horaire'][4].round(1), fontsize=12, color="white", weight="bold")
			plt.text(df_graphe['Comptage horaire'][5]-7, 5+0.1, df_graphe['Comptage horaire'][5].round(1), fontsize=12, color="white", weight="bold")
			plt.text(df_graphe['Comptage horaire'][6]-7, 6+0.1, df_graphe['Comptage horaire'][6].round(1), fontsize=12, color="white", weight="bold")
			plt.text(df_graphe['Comptage horaire'][7]-7, 7+0.1, df_graphe['Comptage horaire'][7].round(1), fontsize=12, color="white", weight="bold")
			st.pyplot(fig)
			st.markdown(			
			"<p style='text-align: justify'>"
			"Toutes les vacances n’ont pas le même effet. "
			"Les vacances de Noël (2019 et 2020) sont les plus impactantes : -43 % par rapport à la moyenne sur la période étudiée. "
			"Puis viennent les vacances de février 2020 avec -33 %. "
			"Les vacances en août et à la Toussaint provoquent une faible baisse, respectivement de -8 % et -5 %. "
			"En revanche, pour le pont de l’Ascension et les vacances en juillet, on observe une hausse significative : +18 % et 34 %. "
			"Nous écartons les vacances de printemps 2020, le trafic étant alors fortement réduit par le premier confinement."
			"</p>", unsafe_allow_html=True)						
		elif select_recur == recur3:
			#st.markdown(select_recur)			
			#insérer codes graphes Météo
			col1, col2, col3 = st.beta_columns([1,2,1])
			with col2:
				df_pluie0 = df_comptages[df_comptages["Date et heure de comptage"].isin(df_date[(df_date["Année"] == 2020) & (df_date["Pluie"] == 0)]["Date et heure de comptage"])]
				df_pluie0["Pluie"] = 0
				df_pluie1 = df_comptages[df_comptages["Date et heure de comptage"].isin(df_date[(df_date["Année"] == 2020) & (df_date["Pluie"] == 1)]["Date et heure de comptage"])]
				df_pluie1["Pluie"] = 1
				df_pluie2 = df_comptages[df_comptages["Date et heure de comptage"].isin(df_date[(df_date["Année"] == 2020) & (df_date["Pluie"] == 2)]["Date et heure de comptage"])]
				df_pluie2["Pluie"] = 2
				df_graphe = pd.concat([df_pluie0, df_pluie1, df_pluie2], ignore_index=True)
				df_graphe = df_graphe.groupby('Pluie', as_index = False).agg({'Comptage horaire':'mean'})
				fig = plt.figure(figsize = (6, 6))
				sns.barplot(x=df_graphe.index, y=df_graphe['Comptage horaire'],palette = 'hls')
				plt.title('Influence de la pluie\n', fontsize = 22)
				plt.ylabel('Nombre moyen de vélos / heure / site', fontsize = 16)
				plt.xticks(range(3), ['Pas de pluie', 'Modérée', 'Forte'], fontsize = 16)
				plt.text(-0.15, 30, df_graphe['Comptage horaire'][0].round(1), fontsize=14, color="white", weight="bold")
				plt.text(0.85, 30, df_graphe['Comptage horaire'][1].round(1), fontsize=14, color="white", weight="bold")
				plt.text(1.85, 30, df_graphe['Comptage horaire'][2].round(1), fontsize=14, color="white", weight="bold")
				st.pyplot(fig)
			st.markdown(			
			"<p style='text-align: justify'>"
			"Sur la période étudiée (septembre 2019 - décembre 2020), la pluie impacte le trafic : -33 % en moyenne. "
			"En revanche, il y a peu de différence entre une pluie modérée et forte. "
			"Les usagers prêts à affronter la pluie le font apparemment coûte que coûte.</p>"
			"<p>Autre facteur météorologique étudié : les températures extrêmes pour Paris.</p>"
			, unsafe_allow_html=True)				
			col1, col2, col3 = st.beta_columns([1,2,1])
			with col2:
				df_froid0 = df_comptages[df_comptages["Date et heure de comptage"].isin(df_date[(df_date["Année"] == 2020) & (df_date["Froid"] == 0)]["Date et heure de comptage"])]
				df_froid0["Froid"] = 0
				df_froid1 = df_comptages[df_comptages["Date et heure de comptage"].isin(df_date[(df_date["Année"] == 2020) & (df_date["Froid"] == 1)]["Date et heure de comptage"])]
				df_froid1["Froid"] = 1
				df_graphe = pd.concat([df_froid0, df_froid1], ignore_index=True)
				df_graphe = df_graphe.groupby('Froid', as_index = False).agg({'Comptage horaire':'mean'})
				fig = plt.figure(figsize = (6, 6))
				sns.barplot(x=df_graphe.index, y=df_graphe['Comptage horaire'],palette = 'hls')
				plt.title('Influence du froid\n', fontsize = 22)
				plt.ylabel('Nombre moyen de vélos / heure / site', fontsize = 16)
				plt.xticks(range(2), ['> 4 °C', '< 4 °C'], fontsize = 16)
				plt.text(-0.12, 25, df_graphe['Comptage horaire'][0].round(1), fontsize=15, color="white", weight="bold")
				plt.text(0.89, 25, df_graphe['Comptage horaire'][1].round(1), fontsize=15, color="white", weight="bold")
				st.pyplot(fig)
			st.markdown(			
			"<p style='text-align: justify'>"
			"Si le thermomètre descend sous les 4 °C, il refroidit 27 % des cyclistes en moyenne."
			, unsafe_allow_html=True)
			col1, col2, col3 = st.beta_columns([1,2,1])
			with col2:	
				df_chaud0 = df_comptages[df_comptages["Date et heure de comptage"].isin(df_date[(df_date["Année"] == 2020) & (df_date["Chaud"] == 0)]["Date et heure de comptage"])]
				df_chaud0["Chaud"] = 0
				df_chaud1 = df_comptages[df_comptages["Date et heure de comptage"].isin(df_date[(df_date["Année"] == 2020) & (df_date["Chaud"] == 1)]["Date et heure de comptage"])]
				df_chaud1["Chaud"] = 1
				df_graphe = pd.concat([df_chaud0, df_chaud1], ignore_index=True)
				df_graphe = df_graphe.groupby('Chaud', as_index = False).agg({'Comptage horaire':'mean'})
				fig = plt.figure(figsize = (6, 6))
				sns.barplot(x=df_graphe.index, y=df_graphe['Comptage horaire'],palette = 'hls')
				plt.title('Influence du beau temps\n', fontsize = 22)
				plt.ylabel('Nombre moyen de vélos / heure / site', fontsize = 16)
				plt.xticks(range(2), ['Autre', '> 25 °C'], fontsize = 16)
				plt.text(-0.12, 40, df_graphe['Comptage horaire'][0].round(1), fontsize=15, color="white", weight="bold")
				plt.text(0.89, 40, df_graphe['Comptage horaire'][1].round(1), fontsize=15, color="white", weight="bold")			
				st.pyplot(fig)
			st.markdown(			
			"<p style='text-align: justify'>"
			"A l’opposé, des températures supérieures à 25°C font sortir, en moyenne, 50 % de cyclistes supplémentaires."
			, unsafe_allow_html=True)						
	#Evènements exceptionnels
	#########################
	elif dataviz_temp == temp4:
		st.subheader(dataviz_temp)
		excep1 = "Grève des transports"
		excep2 = "Covid-19"
		select_excep = st.radio("", (excep1, excep2))
		if select_excep == excep1:
			#Grève
			col1, col2, col3 = st.beta_columns([1, 2, 1])
			with col2:
				df_greve = df_date[(df_date.Année == 2019) | ((df_date.Semaine < 11) & (df_date.Année == 2020))]
				df_greve0 = df_comptages[df_comptages["Date et heure de comptage"].isin(df_greve[df_greve["Grève"] == 0]["Date et heure de comptage"])]
				df_greve0["Grève"] = 0
				df_greve1 = df_comptages[df_comptages["Date et heure de comptage"].isin(df_greve[df_greve["Grève"] == 1]["Date et heure de comptage"])]
				df_greve1["Grève"] = 1
				df_graphe = pd.concat([df_greve0, df_greve1], ignore_index=True)
				df_graphe = df_graphe.groupby('Grève', as_index = False).agg({'Comptage horaire':'mean'})
				fig = plt.figure(figsize = (6, 6))
				sns.barplot(x=df_graphe.index, y=df_graphe['Comptage horaire'],palette = 'hls')
				plt.title('Influence de la grève\n', fontsize = 18)
				plt.ylabel('Nombre moyen de vélos / heure / site', fontsize = 13)
				plt.xticks(range(2), ['avant / après', 'pendant'], fontsize = 13)
				plt.text(-0.12, 32, df_graphe['Comptage horaire'][0].round(1), fontsize=15, color="white", weight="bold")
				plt.text(0.89, 32, df_graphe['Comptage horaire'][1].round(1), fontsize=15, color="white", weight="bold")
				st.pyplot(fig)
			st.markdown(			
			"<p style='text-align: justify'>"
			"Pour distinguer l’effet de la grève (du 5 décembre 2019 au 26 janvier 2020) de celui de la pandémie de Covid-19, "
			"nous avons ici pris en compte la période du 1er septembre 2019 au 16 mars 2020 (veille du premier confinement)."
			"<p style='text-align: justify'>"
			"Pendant la grève des transports de l’hiver 2019/2020, le trafic explose avec +58 % en moyenne."
			"</p><br>"
			, unsafe_allow_html=True)
		elif select_excep == excep2:
			#Covid
			col1, col2, col3 = st.beta_columns([1, 2, 1])
			with col2:
				df_covid = df_date[((df_date.Semaine < 49) & (df_date.Année == 2019)) | ((df_date.Semaine >= 4) & (df_date.Année == 2020))]
				df_covid0 = df_comptages[df_comptages["Date et heure de comptage"].isin(df_covid[df_covid["Covid"] == 0]["Date et heure de comptage"])]
				df_covid0["Covid"] = 0
				df_covid1 = df_comptages[df_comptages["Date et heure de comptage"].isin(df_covid[df_covid["Covid"] == 1]["Date et heure de comptage"])]
				df_covid1["Covid"] = 1
				df_graphe = pd.concat([df_covid0, df_covid1], ignore_index=True)
				df_graphe = df_graphe.groupby('Covid', as_index = False).agg({'Comptage horaire':'mean'})
				fig = plt.figure(figsize = (6, 6))
				sns.barplot(x=df_graphe.index, y=df_graphe['Comptage horaire'],palette = 'hls')
				plt.title('Influence de la pandémie\n', fontsize = 18)
				plt.ylabel('Nombre moyen de vélos par heure', fontsize = 13)
				plt.xticks(range(2), ['Avant Covid-19', 'Covid-19'], fontsize = 13)
				plt.text(-0.12, 30, df_graphe['Comptage horaire'][0].round(1), fontsize=15, color="white", weight="bold")
				plt.text(0.89, 30, df_graphe['Comptage horaire'][1].round(1), fontsize=15, color="white", weight="bold") 
				st.pyplot(fig)
			st.markdown(			
			"<p style='text-align: justify'>"
			"Nous avons retenu le 17 mars 2020 (date du 1er jour de confinement) comme début de la pandémie de Covid-19. "
			"Pour distinguer l’effet de la pandémie de Covid-19 de celui de la grève des transports, "
			"nous avons ici pris en compte la période du 1er septembre 2019 au 31 décembre 2020, hors période de grève (du 5 décembre 2019 au 26 janvier 2020)."
			"<p style='text-align: justify'>"
			"Sur la période étudiée, le trafic augmente de 23 %, en moyenne, après le début de la pandémie."
			"</p>"
			"<p style='text-align: justify'>"
			"<br>Pendant la période Covid-19 il y a eu 3,5 mois de confinement. Quel a été leur effet ?"
			"</p>"
			, unsafe_allow_html=True)				
			col1, col2, col3 = st.beta_columns([1, 2, 1])
			with col2:
				df_confinement = df_date[(df_date.Semaine >= 12) & (df_date.Année == 2020)]
				df_confinement0 = df_comptages[df_comptages["Date et heure de comptage"].isin(df_confinement[df_confinement["Confinement"] == 0]["Date et heure de comptage"])]
				df_confinement0["Confinement"] = 0
				df_confinement1 = df_comptages[df_comptages["Date et heure de comptage"].isin(df_confinement[df_confinement["Confinement"] == 1]["Date et heure de comptage"])]
				df_confinement1["Confinement"] = 1
				df_confinement2 = df_comptages[df_comptages["Date et heure de comptage"].isin(df_confinement[df_confinement["Confinement"] == 2]["Date et heure de comptage"])]
				df_confinement2["Confinement"] = 2
				df_graphe = pd.concat([df_confinement0, df_confinement1, df_confinement2], ignore_index=True)
				df_graphe = df_graphe.groupby('Confinement', as_index = False).agg({'Comptage horaire':'mean'})
				fig = plt.figure(figsize = (6, 6))
				sns.barplot(x=df_graphe.index, y=df_graphe['Comptage horaire'],palette = 'hls')
				plt.title('Influence des confinements\n', fontsize = 18)
				plt.ylabel('Nombre moyen de vélos / heure / site', fontsize = 13)
				plt.xticks(range(3), ['Pas de Confinement', 'Confinement 1', 'Confinement 2'], fontsize = 12, rotation = 20)
				plt.text(-0.15, 35, df_graphe['Comptage horaire'][0].round(1), fontsize=12, color="white", weight="bold")
				plt.text(0.85, 4, df_graphe['Comptage horaire'][1].round(1), fontsize=12, color="white", weight="bold")
				plt.text(1.85, 22, df_graphe['Comptage horaire'][2].round(1), fontsize=12, color="white", weight="bold");
				st.pyplot(fig)
			st.markdown(			
			"<p style='text-align: justify'>"
			"Les deux confinements n’ont pas eu le même impact. "
			"Lors du premier, très strict, le trafic a chuté de 82 % en moyenne contre 36 % seulement pour le deuxième, "
			"aux règles plus souples (écoles et magasins ouverts)."
			"</p>", unsafe_allow_html=True)									
	

########################
## Trafic & accidents ##
########################
elif select_page == page5:
	st.header(select_page)
	acc1 = "Liens entre le trafic et les accidents cyclistes"
	acc2 = "Cartographie des accidents"
	acc3 = "Statistiques sur les personnes accidentées"
	dataviz_acc = st.radio("", (acc1, acc2, acc3))

	# Liens trafic accidents
	if dataviz_acc == acc1:
		st.subheader(dataviz_acc)
		st.markdown(			
		"<p style='text-align: justify'>"
		"Si l'évolution des <strong>comptages horaires par site</strong> reflète l'évolution du trafic cycliste à Paris, "
		"ces comptages en eux-mêmes ne représentent pas la circulation réelle.  "
		"En effet, tous les vélos ne sont pas comptés et certains vélos peuvent être comptés par plusieurs sites durant la même heure. "
		"De plus, le nombre de sites de comptage n'est pas constant lors de la période étudiée (septembre-décembre 2019). "
		"aux règles plus souples (écoles et magasins ouverts). "
		"Nous devons donc utiliser le <span style='color: #1ca2d1; font-weight: bold'>Comptage moyen/site/heure</span>."
		"</p>"
		"<p style='text-align: justify'>"
		"La base de données accidents, en revanche, nous donne le "
		"<span style='color: #1ca2d1; font-weight: bold'>nombre réel d'accidents impliquant des vélos à Paris/heure.</span>"
		"</p>"
		"<p style='text-align: justify'>"
		"<span style='color: #1ca2d1; font-weight: bold'>Ratio Accidents-Trafic</span> = "
		"<span style='color: #1ca2d1'>Nb d'accidents à Paris/heure</span> "
		"/ <span style='color: #1ca2d1'>Comptage moyen/site/heure</span>"
		"</p>"
		, unsafe_allow_html=True)
		#création du df "df_2019_acc"
		df_comptages["Date et heure de comptage"] = pd.to_datetime(df_comptages["Date et heure de comptage"])
		df_comptages['Date'] = df_comptages['Date et heure de comptage'].dt.date
		df_comptages['Année'] = df_comptages['Date et heure de comptage'].dt.year
		df_comptages['Mois'] = df_comptages['Date et heure de comptage'].dt.month
		df_comptages['Jour_de_la_semaine'] = df_comptages['Date et heure de comptage'].dt.weekday
		df_comptages['Semaine'] = df_comptages['Date et heure de comptage'].dt.week
		df_comptages['Jour'] = df_comptages['Date et heure de comptage'].dt.day
		df_comptages_2019 = df_comptages[df_comptages["Année"] == 2019]
		df_comptages_2019 = df_comptages_2019.groupby(['Date et heure de comptage', 'Date', 'Heure', 'Année', 'Mois', 'Jour_de_la_semaine', 'Semaine', 'Jour'], as_index = False).agg({'Comptage horaire':'mean'})
		df_comptages_2019["jour_mois_heure"] = df_comptages_2019["Jour"].map(str) + " / " + df_comptages_2019["Mois"].map(str) + " / " + df_comptages_2019["Heure"].map(str)
		df_acc["heure"] = df_acc["hrmn"].str[:2].astype(int)
		df_acc = df_acc.groupby(["jour", "mois", "an", "heure"], as_index = False).agg({'Num_Acc':'count'})
		df_acc["jour_mois_heure"] = df_acc["jour"].map(str) + " / " + df_acc["mois"].map(str) + " / " + df_acc["heure"].map(str)
		df = df_comptages_2019.merge(df_acc, how = 'left', on = 'jour_mois_heure')
		df_js = df[["jour_mois_heure", "Jour_de_la_semaine"]]
		df_acc = df_acc.merge(df_js, how = 'left', on = 'jour_mois_heure')
		df = df.drop(columns = ['jour', 'mois', 'an', 'jour_mois_heure', 'heure'])
		df = df.fillna(0)
		df_2019_acc = df[["Date", "Mois", "Semaine", "Jour", "Jour_de_la_semaine", "Heure", "Comptage horaire", "Num_Acc"]]
		df_2019_acc = df_2019_acc.rename(columns = {'Comptage horaire':'Comptage_moyen/site/heure','Num_Acc':'Nb_Acc/Paris/heure'})
		df_2019_acc['Ratio Accidents-Trafic'] = df_2019_acc['Nb_Acc/Paris/heure'] / df_2019_acc['Comptage_moyen/site/heure']
		st.dataframe(df_2019_acc.head())
		st.write('Total Nb_Acc :',df_2019_acc['Nb_Acc/Paris/heure'].sum().astype(int))
		st.markdown(			
		"<p style='text-align: justify'>"
		"<strong>L'étude porte sur un petit échantillon (644 accidents, 4 mois), dont voici les tendances. "
		"Elles seront à confirmer sur un plus grand échantillon.</strong>"
		"</p>"
		, unsafe_allow_html=True)
		st.dataframe(df_acc.head())
		st.subheader("ANALYSE PAR HEURE")
		st.markdown("<br><strong>1. Statistiques descriptives</strong>", unsafe_allow_html=True)
		st.dataframe(df_2019_acc.describe())
		st.markdown("<br><strong>2. Distribution de la variable <span style='color: #1ca2d1'>Nombre d’accidents à Paris/heure</span></strong> (fréquence normalisée)", unsafe_allow_html=True)
		# REPARTITION DU NOMBRE D'ACCIDENTS/HEURE
		fig = plt.figure(figsize=(8, 5))
		plt.hist(df_2019_acc['Nb_Acc/Paris/heure'], bins = [0,1,2,3,4,5,6,7,8],rwidth = 0.6, color = '#EE3459', density = True)
		plt.xlabel('Nb accidents/heure')
		plt.ylabel("Fréquence")
		plt.title("Distribution du Nombre d'accidents à Paris/heure")
		st.pyplot(fig)
		st.markdown("<br><strong>3. Nuage de points entre les variables <span style='color: #1ca2d1'>Nombre d’accidents à Paris/heure</span> et <span style='color: #1ca2d1'>Comptage moyen/site/heure</span></strong>", unsafe_allow_html=True)
		# NUAGE DE POINTS POUR COMPARER NB ACCIDENTS & TRAFIC
		fig = plt.figure(figsize=(8,5))
		plt.scatter(df_2019_acc["Comptage_moyen/site/heure"], df_2019_acc['Nb_Acc/Paris/heure'], marker = "H", c = "#338aff", s = 30, alpha = .5)
		plt.xlabel('Comptage moyen/site/heure')
		plt.ylabel("Accidents à Paris/heure")
		plt.title("Accidents & trafic par heure")
		plt.axhline(y = 0.22, color = 'red', label = 'Moyenne accidents', lw = 1)
		plt.legend(loc = 'upper right')
		st.pyplot(fig)
		st.markdown("<br><strong>4. Test ANOVA entre les variables <span style='color: #1ca2d1'>Nombre d’accidents à Paris/heure</span> et <span style='color: #1ca2d1'>Comptage moyen/site/heure</span></strong>", unsafe_allow_html=True)
		df_2019_acc_a = df_2019_acc.rename(columns = {'Nb_Acc/Paris/heure':'Acc','Comptage_moyen/site/heure':'Comptage'})
		result = statsmodels.formula.api.ols('Acc ~ Comptage', data = df_2019_acc_a).fit()
		table = statsmodels.api.stats.anova_lm(result)
		st.dataframe(table)
		st.markdown(			
		"<p style='text-align: justify'>"
		"<strong>Conclusion</strong><br>"
		"Lors de la période étudiée :"
		"<ul>"
		  "<li>En moyenne, on observe <span style='color: #1ca2d1'><strong>0,22 accidents de vélos/heure à Paris</span></strong>.</li>"
		  "<li>Dans 90% des cas, il n’y a pas d’accident.</li>"
		  "<li><span style='color: #1ca2d1'><strong>Dans 8 % des cas, il y a 1 à 2 accidents de vélos/heure</span></strong>.</li>"
		  "<li>La p-value du test ANOVA est inférieure à 5%, on rejette donc l'hypothèse selon laquelle le <strong>comptage moyen/site/heure</strong> n'influe pas sur le <strong>nombre d'accidents à Paris/heure</strong>.</li>"
		"</ul>"
		"</p>"
		, unsafe_allow_html=True)
		st.markdown("<br><strong>5. Distribution du <span style='color: #1ca2d1'>Ratio Accidents-Trafic</span> en fonction de l’heure</strong>", unsafe_allow_html=True)
		df_comptages_2019_h = df_comptages[df_comptages["Année"] == 2019]
		df_comptages_2019_h = df_comptages_2019_h.groupby(['Heure'], as_index = False).agg({'Comptage horaire':'mean'})
		df_acc_h = df_acc.rename(columns = {'heure':'Heure'})
		df_acc_h = df_acc_h.groupby(["Heure"], as_index = False).agg({'Num_Acc':'sum'})
		df = df_comptages_2019_h.merge(df_acc_h, how = 'left', on = 'Heure')
		df = df.fillna(0)
		df_ratio_h = df[["Heure", "Comptage horaire", "Num_Acc"]]
		df_ratio_h = df_ratio_h.rename(columns = {'Comptage horaire':'Comptage_moyen/site/heure','Num_Acc':'Nb_Acc/Paris/heure'})
		df_ratio_h['Ratio Accidents-Trafic'] = df_ratio_h['Nb_Acc/Paris/heure'] / df_ratio_h['Comptage_moyen/site/heure']
		med = df_ratio_h["Ratio Accidents-Trafic"].median()
		moy = df_ratio_h["Ratio Accidents-Trafic"].mean()
		# REPARTITION DU RATIO EN FONCTION DE L'HEURE (BARPLOT)
		fig = plt.figure(figsize=(14, 8))
		ax1 = fig.add_subplot()
		ax1.set_xlabel('\nHeure', fontsize=15)
		ax1.set_ylabel("Ratio Accidents-Trafic", fontsize=15)
		ax1.bar(df_ratio_h['Heure'], df_ratio_h['Ratio Accidents-Trafic'])
		ax1.set_title("Ratio Accidents-Trafic par heure", fontsize=20)
		ax1.yaxis.set_tick_params(width = 2, length = 10, labelsize = 10)
		ax1.xaxis.set_tick_params(width = 2, length = 10, labelsize = 10)
		ax1.axhline(y = moy, xmin = 0.05, xmax = 0.95, color = 'red', label = 'Moyenne', lw = 4)
		ax1.axhline(y = med, xmin = 0.05, xmax = 0.95, color = 'yellow', label = 'Médiane', lw = 4)
		ax1.legend(fontsize = 15)
		plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23], ['minuit',' ',' ',' ','4h',' ',' ',' ','8h',' ',' ',' ','midi',' ',' ',' ','16h',' ',' ',' ','20h',' ',' ','23h'])
		fig.tight_layout()
		st.pyplot(fig)
		st.markdown(			
		"<p style='text-align: justify'>"
		"<strong>Conclusion</strong><br>"
		"<ul>"
		  "<li><span style='color: #1ca2d1'><strong>En journée, les heures de pointe (8-9h et 18-19h) ont un ratio Accidents-Trafic sous la moyenne, ce qui en fait des heures peu accidentogènes</strong></span>. "
		  "Les usagers utilisant le vélo quotidiennement pour aller travailler seraient-ils plus aguerris ?</li>"
		  "<li><span style='color: #1ca2d1'><strong>La nuit, on observe les ratios les plus élevés. 3 heures est l’heure la plus accidentogène (7 fois la moyenne)</strong></span> : "
		  "A Paris, cela correspond à l’horaire de fermeture des bars. "
		  "<span style='color: #1ca2d1'><strong>A 6 heures, on observe un second pic (2 fois la moyenne)</strong></span>. "
		  "Le manque de visibilité et la conduite en état d'ivresse sont des pistes à étudier.</li>"
		"</ul>"
		"</p><br>"
		, unsafe_allow_html=True)
		st.subheader("ANALYSE PAR JOUR")
		st.markdown("<br><strong>1. Nuage de points entre les variables <span style='color: #1ca2d1'>Nombre d’accidents à Paris/jour</span> et "
			"<span style='color: #1ca2d1'>Comptage moyen/site/jour</span></strong>", unsafe_allow_html=True)
		# NUAGE DE POINTS POUR COMPARER NB ACCIDENTS & TRAFIC
		df_2019_acc_date = df_2019_acc.groupby('Date', as_index=False).agg({'Comptage_moyen/site/heure':'sum','Nb_Acc/Paris/heure':'sum'})
		df_2019_acc_date['Ratio Accidents-Trafic'] = df_2019_acc_date['Nb_Acc/Paris/heure'] / df_2019_acc_date['Comptage_moyen/site/heure']
		df_2019_acc_date = df_2019_acc_date.rename(columns = {'Comptage_moyen/site/heure':'Comptage_moyen/site/jour', 'Nb_Acc/Paris/heure':'Nb_Acc/Paris/jour'})
		fig = plt.figure(figsize=(8,5))
		plt.scatter(df_2019_acc_date["Comptage_moyen/site/jour"], df_2019_acc_date['Nb_Acc/Paris/jour'], marker = "H", c = "#338aff", s = 30, alpha = .5)
		plt.xlabel('Comptage moyen/site/jour')
		plt.ylabel("Accidents à Paris/jour")
		plt.title("Accidents & trafic par jour")
		plt.axhline(y = 5.3, color = 'red', label = 'Moyenne accidents', lw = 1)
		plt.legend(loc = 'upper left')
		st.pyplot(fig)
		st.markdown(			
		"<p style='text-align: justify'>"
		"Lors de la période étudiée :"
		"<ul>"
		  "<li>En moyenne, on observe <span style='color: #1ca2d1'><strong>5,3 accidents de vélos/jour à Paris</span></strong>.</li>"
		  "<li><span style='color: #1ca2d1'><strong>Dans 79% des cas, il y a entre 0 et 7 accidents/jour</span></strong>.</li>"
		  "<li>Dans 95%, il y a moins de 12 accidents/jour, le maximum étant 19.</li>"
		"</ul>"
		"</p>"
		, unsafe_allow_html=True)
		st.markdown("<br><strong>2. Distribution du <span style='color: #1ca2d1'>Ratio Accidents-Trafic</span> en fonction du jour</strong>", unsafe_allow_html=True)
		df_comptages_2019 = df_comptages[df_comptages["Année"] == 2019]
		df_comptages_2019 = df_comptages_2019.groupby(['Date', 'Mois', 'Jour'], as_index = False).agg({'Comptage horaire':'mean'})
		df_comptages_2019["jour_mois"] = df_comptages_2019["Jour"].map(str) + " / " + df_comptages_2019["Mois"].map(str)
		df_acc_j = df_acc.groupby(["jour", "mois"], as_index = False).agg({'Num_Acc':'count'})
		df_acc_j["jour_mois"] = df_acc_j["jour"].map(str) + " / " + df_acc_j["mois"].map(str)
		df = df_comptages_2019.merge(df_acc_j, how = 'left', on = 'jour_mois')
		df = df.drop(columns = ['jour', 'mois','jour_mois'])
		df = df.fillna(0)
		df_ratio_j = df[["Date", "Comptage horaire", "Num_Acc"]]
		df_ratio_j = df_ratio_j.rename(columns = {'Comptage horaire':'Comptage_moyen/site/heure','Num_Acc':'Nb_Acc/Paris/heure'})
		df_ratio_j['Ratio Accidents-Trafic'] = df_ratio_j['Nb_Acc/Paris/heure'] / df_ratio_j['Comptage_moyen/site/heure']
		med = df_ratio_j["Ratio Accidents-Trafic"].median()
		moy = df_ratio_j["Ratio Accidents-Trafic"].mean()
		fig = plt.figure(figsize=(14, 8))
		ax1 = fig.add_subplot()
		ax1.set_xlabel('\nJour', fontsize=15)
		ax1.set_ylabel("Ratio Accidents-Trafic", fontsize=15)
		ax1.bar(df_ratio_j['Date'], df_ratio_j['Ratio Accidents-Trafic'])
		ax1.set_title("Ratio Accidents-Trafic par jour", fontsize=20)
		ax1.yaxis.set_tick_params(width = 2, length = 10, labelsize = 10)
		ax1.xaxis.set_tick_params(width = 2, length = 10, labelsize = 10)
		ax1.axhline(y = moy, xmin = 0.05, xmax = 0.95, color = 'red', label = 'Moyenne', lw = 4)
		ax1.axhline(y = med, xmin = 0.05, xmax = 0.95, color = 'yellow', label = 'Médiane', lw = 4)
		ax1.legend(fontsize = 15)
		fig.tight_layout()
		st.pyplot(fig)
		st.markdown(			
		"<p style='text-align: justify'>"
		"<ul>"
		  "<li>Il existe une forte variabilité du ratio Accidents-Trafic selon les jours.</li>"
		  "<li>On n’observe pas de périodicité hebdomadaire (jours de semaine vs week-end).</li>"
		  "<li><span style='color: #1ca2d1'><strong>La grève des transports en décembre, où le trafic cycliste a pourtant explosé (+58 %), est peu accidentogène</span></strong> : "
		  "la plupart des jours sont sous la moyenne et les pics moins nombreux. Serait-ce aussi lié au profil des usagers ?</li>"
		"</ul>"
		"</p>"
		, unsafe_allow_html=True)
		st.subheader("ANALYSE PAR JOUR DE LA SEMAINE")
		st.markdown("<br><strong>Distribution du <span style='color: #1ca2d1'>Ratio Accidents-Trafic</span> en fonction du jour de la semaine</strong>", unsafe_allow_html=True)
		df_comptages_2019 = df_comptages[df_comptages["Année"] == 2019]
		df_comptages_2019 = df_comptages_2019.groupby(["Jour_de_la_semaine"], as_index = False).agg({'Comptage horaire':'mean'})
		df_acc_j = df_acc.groupby(["Jour_de_la_semaine"], as_index = False).agg({'Num_Acc':'count'})
		df = df_comptages_2019.merge(df_acc_j, how = 'left', on = 'Jour_de_la_semaine')
		df = df.fillna(0)
		df_ratio_js = df[["Jour_de_la_semaine", "Comptage horaire", "Num_Acc"]]
		df_ratio_js = df_ratio_js.rename(columns = {'Comptage horaire':'Comptage_moyen/site/heure','Num_Acc':'Nb_Acc/Paris/heure'})
		df_ratio_js['Ratio Accidents-Trafic'] = df_ratio_js['Nb_Acc/Paris/heure'] / df_ratio_js['Comptage_moyen/site/heure']
		med = df_ratio_js["Ratio Accidents-Trafic"].median()
		moy = df_ratio_js["Ratio Accidents-Trafic"].mean()
		fig = plt.figure(figsize=(14, 8))
		ax1 = fig.add_subplot()
		ax1.set_xlabel('\nJour_de_la_semaine', fontsize=15)
		ax1.set_ylabel("Ratio Accidents-Trafic", fontsize=15)
		ax1.bar(df_ratio_js['Jour_de_la_semaine'], df_ratio_js['Ratio Accidents-Trafic'])
		ax1.set_title("Ratio Accidents-Trafic par jour de la semaine", fontsize=20)
		ax1.yaxis.set_tick_params(width = 2, length = 10, labelsize = 10)
		ax1.xaxis.set_tick_params(width = 2, length = 10, labelsize = 10)
		ax1.axhline(y = moy, xmin = 0.05, xmax = 0.95, color = 'red', label = 'Moyenne', lw = 4)
		ax1.axhline(y = med, xmin = 0.05, xmax = 0.95, color = 'yellow', label = 'Médiane', lw = 4)
		ax1.legend(fontsize = 15)
		plt.xlabel('Jour de la semaine')
		plt.xticks([0,1,2,3,4,5,6], ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'], fontsize=13)
		fig.tight_layout()
		st.pyplot(fig)
		st.markdown(
		"<p style='text-align: justify'>"
		"<ul>"
		  "<li>Le ratio Accident-Trafic est relativement stable dans la semaine.</li>"
		  "<li><span style='color: #1ca2d1'><strong>Les journées les moins accidentogènes sont le dimanche et</span></strong>, "
		  "dans une moindre mesure, <span style='color: #1ca2d1'><strong>le jeudi</span></strong>.</li>"
		  "<li><span style='color: #1ca2d1'><strong>La journée la plus accidentogène est le mercredi</span></strong>.</li>"
		"</ul>"
		"</p>"
		, unsafe_allow_html=True)



	#cartographie accidents
	elif dataviz_acc == acc2:
		st.subheader(dataviz_acc)
		df_acc["lat"] = df_acc["lat"].astype(float)
		df_acc["long"] = df_acc["long"].astype(float)
		st.markdown('Sélectionnez les éléments à afficher sur la carte :')
		col1, col2 = st.beta_columns([2.7, 2.3])
		with col1:
			legers = st.checkbox("Accidents avec cycliste indemne ou blessé léger", value=False)
			hosp = st.checkbox("Accidents avec cycliste blessé et hospitalisé", value=False)
			tues = st.checkbox("Accidents avec cycliste tué", value=False)
		with col2:
			pcycl = st.checkbox("Accidents survenus sur piste cyclable", value=False)
			hors_pcycl = st.checkbox("Accidents survenus hors piste cyclable", value=False)
			trafic = st.checkbox("Sites de comptages", value=False)
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
			for nom, comptage, latitude, longitude, image in zip(df_2019["Nom du site de comptage"],
			                                                     df_2019["Comptage horaire"],
			                                                     df_2019["lat"],
			                                                     df_2019["long"],
			                                                     df_2019["Lien vers photo du site de comptage"]):
				pp = "<strong>" + nom + "</strong>" + "<br>Comptage horaire : " + str(round(comptage,2)) + "<br><img src='" + image + "', width=100%>"
				folium.CircleMarker(location=[latitude, longitude],radius=comptage/8,popup = folium.Popup(pp, max_width = 300),tooltip= "<strong>" + nom + "</strong>",color = "#368fe5",fill_color = "#368fe5",fill_opacity=0.3).add_to(carte)
		folium_static(carte)
		#commentaires carte		
		st.markdown(body=
		"<strong><font size=20><span style='color:green'>.</span></font></strong>   accident avec blessé léger  - "
		"<strong><font size=20><span style='color:orange'>.</span></font></strong>   accident avec blessé hospitalisé  - "
		"<strong><font size=20><span style='color:red'>.</span></font></strong>   accident avec décés<br>"
		"<strong><font size=5><span style='color:yellow'>o</span></font></strong>  accident sur piste cyclable  - "
		"<strong><font size=5><span style='color:black'>o</span></font></strong>  accident hors piste cyclable", unsafe_allow_html=True)
		st.markdown("<p style='text-align: justify'>"
		"<strong>Les accidents sont plus nombreux aux grands carrefours</strong> (Opéra, Saint Lazare, Gare de l'Est, Châtelet) "
		"<strong>et sur les grands axes</strong> (Sébastopol, Convention, Lafayette, Belleville), pourtant équipés de pistes cyclables. "
		"Idem sur le boulevard des Maréchaux, dans le sud de Paris. "
		"L'explication est à chercher du côté des véhicules en circulation, bien plus nombreux à ces endroits."
		"</p>"
		"<p style='text-align: justify'>"
		"Il y a eu des accidents partout, mais <strong>l'est parisien est plus touché, par rapport aux 7e et 16e arrondissements</strong> par exemple, "
		"où il y a eu le moins d'accidents. Cela reflète la proportion d'usagers du vélo. "
		"A l'est, les quartiers sont plus jeunes et plus animés que dans l’ouest parisien."
		"</p>"
		, unsafe_allow_html=True)
	#Statistiques personnes accidentées
	elif dataviz_acc == acc3:
		st.subheader(dataviz_acc)
		#intitulés de la liste de choix :
		diag1 = "du sexe / du niveau de gravité"
		diag2 = "de l'âge"
		diag3 = "du trajet"
		diag4 = "du type de voie"
		diag5 = "de la météo"
		select_diag = st.radio("Cyclistes accidentés en fonction :", (diag1, diag2, diag3, diag4, diag5))
		#accidentés par sexe
		####################
		if select_diag == diag1:
			#option1_diag1 = "1. Cyclistes accidentés selon leur sexe"
			#option2_diag1 = "2. Cyclistes accidentés selon le niveau gravité"
			#select_diag1 = st.selectbox('', (option1_diag1, option2_diag1), index = 0)
			#graphe sexe
			#if select_diag1 == option1_diag1:
			col1, col2, col3 = st.beta_columns([1, 2, 1])
			with col2:
				plt.rcParams['font.size'] = 7
				fig = plt.figure(figsize=(3, 3))
				plt.title("Cyclistes accidentés selon le sexe", fontsize=9)
				df_sex = df_acc.groupby('sexe', as_index = True).agg({'sexe':'count'})
				plt.pie(x = df_sex.sexe,
				        labels = ['Homme', 'Femme'],
				        explode = [0, 0],
				        autopct = lambda x : str(round(x, 2)) + '%',
				        pctdistance = 0.6,
				        labeldistance = 1.1,
				        shadow = False,
				        radius= 1.1);
				st.pyplot(fig)
			st.markdown("<p style='text-align: justify'>Parmi les cyclistes accidentés, 74 % d’hommes et 26 % de femmes.</p>"
			, unsafe_allow_html=True)
			#graphe hommes/femmes et gravité
			col1, col2 = st.beta_columns(2)
			with col1 :
				plt.rcParams['font.size'] = 8
				fig = plt.figure(figsize=(4, 4))
				plt.title("Gravité pour les hommes", fontsize=12)
				df_hom = df_acc[df_acc['sexe'] == 1]
				df_hom = df_hom.groupby('grav', as_index = True).agg({'grav':'count'})
				df_hom.loc[5,:] = df_hom.loc[2,:]
				df_hom = df_hom.drop(df_hom.index[1])
				plt.pie(x = df_hom.grav,
				        labels = ['Indemne ', 'Blessé léger', 'Blessé hospitalisé', 'Tué'],
				        explode = [0, 0, 0, 0],
				        autopct = lambda x : str(round(x, 2)) + '%',
				        pctdistance = 0.7,
				        labeldistance = 1.1,
				        shadow = False,
				        colors = ("lightgreen", "lightyellow", "orange", "red"))
				st.pyplot(fig)
			with col2 :					
				fig = plt.figure(figsize=(4, 4))
				plt.title("Gravité pour les femmes", fontsize=12)
				df_fem = df_acc[df_acc['sexe'] == 2]
				df_fem = df_fem.groupby('grav', as_index = True).agg({'grav':'count'})
				plt.pie(x = df_fem.grav,
				        labels = ['Indemne ', 'Blessé hospitalisé', 'Blessé léger'],
				        explode = [0, 0, 0],
				        autopct = lambda x : str(round(x, 2)) + '%',
				        pctdistance = 0.7,
				        labeldistance = 1.1,
				        shadow = False,
				        colors = ("lightgreen", "orange", "lightyellow"),
				        startangle=45);
				st.pyplot(fig)
			st.markdown("<p style='text-align: justify'>Pour les hommes, il y a environ la moitié des accidents sans blessure et une autre moitié "
			"avec blessures légères. Pour les femmes, c’est plutôt 1/4 d’accidents sans blessure et 3/4 "
			"avec des blessures légères. Pour les deux, la proportion d'accidents entraînant une "
			"hospitalisation est assez faible, d’environ 2 %. Sur la période (4 mois et 644 accidents), il n’y "
			"a eu qu’un seul mort, un homme.</p>"
			, unsafe_allow_html=True)
			#accidentés par âge
			####################
		elif select_diag == diag2:
			#graphe age
			col1, col2, col3 = st.beta_columns([.5, 3, .5])
			with col2:
				fig = plt.figure(figsize=(3, 3))
				plt.rcParams['font.size'] = 6
				plt.title("Cyclistes accidentés selon l'âge", fontsize=9)
				df_age = df_acc.groupby('age', as_index = False).agg({'Num_Acc':'count'})
				bins = pd.IntervalIndex.from_tuples([(0, 12), (12, 18), (18, 30), (30, 40), (40, 50), (60,70), (70,150)])
				df_age["cat_age"] = pd.cut(df_age["age"], bins)
				df_age = df_age.groupby('cat_age', as_index = True).agg({'Num_Acc':'sum'})
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
				        pctdistance = 0.7,
				        labeldistance = 1.1,
				        shadow = False);
				st.pyplot(fig)
			st.markdown("<p style='text-align: justify'>Sans surprise la majorité des accidentés ont entre 18 et 30 ans, ce qui correspond à la tranche "
			"d’âge qui roule le plus à vélo. Puis le nombre décroît avec l’âge (et l’usage).</p>", unsafe_allow_html=True)
		#accidentés par trajet
		####################
		elif select_diag == diag3:
			#graphe trajet
			col1, col2, col3 = st.beta_columns([.5, 3, .5])
			with col2:
				fig = plt.figure(figsize=(3, 3))
				plt.rcParams['font.size'] = 5
				plt.title("Cyclistes accidentés selon la nature du trajet", fontsize=9)
				df_traj = df_acc.groupby('trajet', as_index = True).agg({'trajet':'count'})
				df_traj.loc[10,:] = df_traj.loc[2,:]
				df_traj.loc[11,:] = df_traj.loc[9,:]
				df_traj = df_traj.drop(df_traj.index[2])
				df_traj = df_traj.drop(df_traj.index[-3])
				plt.pie(x = df_traj.trajet,
				        labels = ['Non renseigné',
				                  'Domicile – travail',
				                  'Courses – achats ',
				                  'Utilisation professionnelle',
				                  'Promenade – loisirs',
				                  'Domicile – école',
				                  'Autre'],
				        explode = [0, 0, 0, 0, 0, 0, 0],
				        autopct = lambda x : str(round(x, 2)) + '%',
				        pctdistance = 0.7,
				        labeldistance = 1.1,
				        shadow = False,
				        startangle = 90)
				st.pyplot(fig)
			st.markdown("<p style='text-align: justify'>Avec 1/3 des trajets non renseignés, difficile de conclure, même si les trajets "
			"Promenade/loisirs et Domicile-travail semblent largement en tête.</p>", unsafe_allow_html=True)
		#accidentés par voie
		####################
		elif select_diag == diag4:
			#graphe situation
			fig = plt.figure(figsize=(3, 3))
			col1, col2, col3 = st.beta_columns([.5, 3, .5])
			with col2:
				plt.rcParams['font.size'] = 6
				plt.title("Cyclistes accidentés selon la voie utilisée", fontsize=9)
				df_situ = df_acc.groupby('situ', as_index = True).agg({'situ':'count'})
				plt.pie(x = df_situ.situ,
				        labels = ['Sur chaussée',
				                  'Sur trottoir',
				                  'Sur piste cyclable',
				                  'Sur autre voie spéciale'],
				        explode = [0, 0, 0, 0],
				        autopct = lambda x : str(round(x, 2)) + '%',
				        pctdistance = 0.7,
				        labeldistance = 1.1,
				        shadow = False,
				        startangle = 90);
				st.pyplot(fig)
			st.markdown("<p style='text-align: justify'>54% des accidents ont eu lieu sur la chaussée contre 37 % sur une piste cyclable."
			"</p>", unsafe_allow_html=True)
		#graphe météo
		elif select_diag == diag5:
			col1, col2, col3 = st.beta_columns([.5, 2, .5])
			with col2:
				fig = plt.figure(figsize=(3, 3))
				plt.rcParams['font.size'] = 5
				plt.title("Cyclistes accidentés selon la météo", fontsize=9)
				df_atm = df_acc.groupby('atm', as_index = True).agg({'atm':'count'})
				df_atm.loc[7,:] = df_atm.loc[7,:] + 1
				df_atm = df_atm.drop(df_atm.index[3])
				df_atm.loc[9,:] = df_atm.loc[7,:]
				df_atm = df_atm.drop(df_atm.index[-3])
				plt.pie(x = df_atm.atm,
				        labels = ['Normale',
				                  'Pluie légère',
				                  'Pluie forte',
				                  'Temps couvert',
				                  'Brouilard'],
				        autopct = lambda x : str(round(x, 2)) + '%',
				        pctdistance = 0.7,
				        labeldistance = 1.1,
				        shadow = False)
				st.pyplot(fig)
			st.markdown("<p style='text-align: justify'>Les 3/4 des accidents ont lieu sous une météo normale. Le 1/4 restant a lieu sous la pluie ou "
			"par temps couvert.</p>", unsafe_allow_html=True)
###########################
## Prédictions comptages ##
###########################
elif select_page == page6:
	st.header(select_page)
	@st.cache(suppress_st_warning=True, max_entries=50, ttl=120)
	def evaluation(variables, algo, taille_test, standardisation):
	    """
	    Fonction qui retourne pour un modèle de régression le score R² sur l'échantillon d'entraînement,
	    le score R² sur l'échantillon de test, la rmse sur l'échantillon d'entraîenement,
	    la rmse sur l'échantillon de test, les prédictions sur l'échantillon d'entraînement et
	    les prédictions sur l'échantillon de test
	    Paramètres :
	    variables (liste) : liste des variables (colonnes) du df à prendre dans l'échantillon d'entraînement et de test
	    algo (string) : nom de l'algorithme parmi la liste suivante : ['LinearRegression',
	                                                                    'Ridge',
	                                                                    'Lasso',
	                                                                    'ElasticNet',
	                                                                    'DecisionTreeRegressor*',
	                                                                    'RandomForestRegressor*',
	                                                                    'BaggingRegressor*',
	                                                                    'GradientBoostingRegressor*']
	    taille_test (float) : taille de l'échantillon de test
	    standardisation (boolean) : True pour standardiser, sinon False
	    """
	    data = df_ml[liste_var]
	    target = df_ml['Comptage_horaire']
	    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = taille_test, shuffle = False)
	    if algo == 'LinearRegression':
	        modele = LinearRegression()
	    elif algo == 'Ridge':
	        modele = RidgeCV(alphas = (0.001, 0.01, 0.1, 0.3, 0.7, 1, 10, 50, 100))
	    elif algo == 'Lasso':
	        modele = LassoCV(alphas = (0.001, 0.01, 0.1, 0.3, 0.7, 1, 10, 50, 100))
	    elif algo == 'ElasticNet':
	        modele = ElasticNet()
	    elif algo == 'DecisionTreeRegressor*':
	        modele = DecisionTreeRegressor()
	    elif algo == 'RandomForestRegressor*':
	        modele = RandomForestRegressor(n_estimators = 10, criterion = 'mse')
	    elif algo == 'BaggingRegressor*':
	        modele = BaggingRegressor(n_estimators = 10)
	    else :
	        modele = GradientBoostingRegressor(n_estimators = 100)
	    if standardisation:
	        scaler = preprocessing.StandardScaler().fit(X_train) 
	        X_train_scaled = scaler.transform(X_train)
	        X_test_scaled = scaler.transform(X_test)
	        modele.fit(X_train_scaled, y_train)
	        pred_train_ = modele.predict(X_train_scaled)
	        pred_test_ = modele.predict(X_test_scaled)
	        r2_train_ = modele.score(X_train_scaled, y_train)
	        r2_test_ = modele.score(X_test_scaled, y_test)
	        rmse_train_ = np.sqrt(mean_squared_error(y_train, pred_train_))
	        rmse_test_ = np.sqrt(mean_squared_error(y_test, pred_test_))
	    else:
	        modele.fit(X_train, y_train)
	        pred_train_ = modele.predict(X_train)
	        pred_test_ = modele.predict(X_test)
	        r2_train_ = modele.score(X_train, y_train)
	        r2_test_ = modele.score(X_test, y_test)
	        rmse_train_ = np.sqrt(mean_squared_error(y_train, pred_train_))
	        rmse_test_ = np.sqrt(mean_squared_error(y_test, pred_test_))
	    return r2_train_, r2_test_, rmse_train_, rmse_test_, pred_test_, pred_train_

	label1="Derniers jours du mois"
	label2="Derniers mois de la période"
	select_pred_ml = st.radio(
	    "Sélectionnez la période à prédire :",
	    (label1,
	    label2))

	#Prédiction sur les derniers jours de chaque mois
	#################################################
	if select_pred_ml == label1:
		st.subheader("Prédictions sur les derniers jours du mois")

		st.markdown('<br><strong>Sélection des paramètres du modèle</strong><br>', unsafe_allow_html=True)

		df_ml = df_ml.sort_values(by = ['Jour'])

		liste_var = st.multiselect('Sélectionnez les variables :',
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
		st.markdown("<br>", unsafe_allow_html=True)

		col1, col2, col3, col4 = st.beta_columns([1, 0.1, 0.9, 0.1])
		with col3:
			jour_deb_test = st.slider(label = "Sélectionnez le premier jour de prédictions :",
			                          min_value = 20, max_value = 31, step = 1, value = 24)
			@st.cache(suppress_st_warning=True, max_entries=5, ttl=120)
			def calculating_test_size(jour):
				##calcul du % à prendre pour la taille de l'échantillon test :
				#nb de lignes df_ml
				len_df_ml = len(df_ml)
				#nb de lignes échantillon test
				len_df_test = len(df_ml[df_ml["Jour"] >= jour])
				test_size_ = len_df_test / len_df_ml
				test_size_round_ = round(test_size_*100, 2)
				return test_size_, test_size_round_
			test_size, test_size_round = calculating_test_size(jour_deb_test)



			st.markdown("<br>", unsafe_allow_html=True)		
			st.write("Taille échantillon de test = ", (test_size_round), " %")
			st.markdown("<br>", unsafe_allow_html=True)
			standardiser = st.checkbox("Standardiser les données", value=False)		
		with col1:
			data = df_ml[liste_var]
			target = df_ml['Comptage_horaire']
			X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = test_size, shuffle = False)
			select_algo = st.radio(
			    "Sélectionnez l'algorithme à tester :",
			    ('LinearRegression',
			    'Ridge',
			    'Lasso',
			    'ElasticNet',
			    'DecisionTreeRegressor*',
			    'RandomForestRegressor*',
			    'BaggingRegressor*',
			    'GradientBoostingRegressor*'))
			st.markdown('<p>* le calcul peut prendre plusieurs minutes</p>', unsafe_allow_html=True)

		st.markdown("<br><strong>Score du modèle choisi</strong>", unsafe_allow_html=True)
		r2_train, r2_test, rmse_train, rmse_test, pred_test, pred_train = evaluation(liste_var, select_algo, test_size, standardiser)
		st.write(select_algo, " :")
		arrondi = 3
		st.write("score R² train = ", round(r2_train, arrondi), " / score R² test = ", round(r2_test, arrondi))
		st.write("rmse train = ", round(rmse_train, arrondi), " / rmse test = ", round(rmse_test, arrondi))


		#Définition dumois et du site représenter les prévisions :		
		st.markdown('<br><strong>Représentation graphique des prédictions</strong><br>', unsafe_allow_html=True)
		#Définition du site et du mois pour représenter les prévisions :
		col1, col2= st.beta_columns(2)
		with col1:
			liste_sites = sorted(df_ml['Nom du site de comptage'].unique().tolist())
			site = st.selectbox('Sélectionnez le site de comptage :', (liste_sites), index = 3)
		with col2:
			select_mois = st.selectbox(
			    "Sélectionnez le mois :",
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
	elif select_pred_ml == label2:
		st.subheader("Prédictions sur les derniers mois de la période")

		st.markdown('<br><strong>Sélection des paramètres du modèle</strong><br>', unsafe_allow_html=True)

		df_ml = df_ml.sort_values(by = ['Année', 'Mois', 'Jour'])

		liste_var = st.multiselect('Sélectionnez les variables :',
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
		st.markdown("<br>", unsafe_allow_html=True)
		col1, col2, col3, col4 = st.beta_columns([1, 0.1, 0.9, 0.1])
		with col3:
			select_mois_deb_test = st.radio("Sélectionnez le premier mois des prédictions :",
										    ('Septembre 2020',
										    'Octobre 2020',
										    'Novembre 2020',
										    'Décembre 2020'), index = 1)
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
			annee_deb_test = 2020
			@st.cache(suppress_st_warning=True, max_entries=5, ttl=120)
			def calculating_test_size_2(annee, mois):
				##calcul du % à prendre pour la taille de l'échantillon test :
				#nb de lignes df_ml
				len_df_ml = len(df_ml)
				#nb de lignes échantillon test
				len_df_test2 = len(df_ml[(df_ml["Mois"] >= mois) & (df_ml["Année"] >= annee)])
				test_size2_ = len_df_test2 / len_df_ml
				test_size_round2_ = round(test_size2_*100, 2)
				return test_size2_, test_size_round2_
			test_size2, test_size_round2 = calculating_test_size_2(annee_deb_test, mois_deb_test)

			if mois_deb_test == 12:
				st.write("Période de prédictions :<br>Décembre 2020", unsafe_allow_html=True)
			else :
				st.write("Période de prédictions :<br>De ", select_mois_deb_test, "à Décembre 2020", unsafe_allow_html=True)

			st.write("Taille échantillon de test = ", (test_size_round2), " %")
			standardiser = st.checkbox("Standardiser les données", value=False)
		with col1:
			data = df_ml[liste_var]
			target = df_ml['Comptage_horaire']
			X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = test_size2, shuffle = False)
			select_algo = st.radio(
			    "Sélectionnez l'algorithme à tester :",
			    ('LinearRegression',
			    'Ridge',
			    'Lasso',
			    'ElasticNet',
			    'DecisionTreeRegressor*',
			    'RandomForestRegressor*',
			    'BaggingRegressor*',
			    'GradientBoostingRegressor*'))
			st.markdown('<p>* le calcul peut prendre plusieurs minutes</p>', unsafe_allow_html=True)
		st.markdown("<br><strong>Score du modèle choisi</strong>", unsafe_allow_html=True)
		r2_train, r2_test, rmse_train, rmse_test, pred_test, pred_train = evaluation(liste_var, select_algo, test_size2, standardiser)
		st.write(select_algo, " :")
		arrondi = 3
		st.write("score R² train = ", round(r2_train, arrondi), " / score R² test = ", round(r2_test, arrondi))
		st.write("rmse train = ", round(rmse_train, arrondi), " / rmse test = ", round(rmse_test, arrondi))

		#Définition du mois et du site représenter les prévisions :
		st.markdown('<br><strong>Représentation graphique des prédictions</strong><br>', unsafe_allow_html=True)
		col1, col2= st.beta_columns(2)
		with col1:	
			liste_sites = sorted(df_ml['Nom du site de comptage'].unique().tolist())
			site = st.selectbox('Sélectionnez le site de comptage :', (liste_sites), index = 3)
		with col2:
			if mois_deb_test == 9:
				select_mois = st.selectbox(
			    "Sélectionnez le mois :",
			    ('Septembre 2020',
			    'Octobre 2020',
			    'Novembre 2020',
			    'Décembre 2020'))
			elif mois_deb_test == 10:
				select_mois = st.selectbox(
			    "Sélectionnez du mois :",
			    ('Octobre 2020',
			    'Novembre 2020',
			    'Décembre 2020'))
			elif mois_deb_test == 11:
				select_mois = st.selectbox(
			    "Sélectionnez du mois :",
			    ('Novembre 2020',
			    'Décembre 2020'))
			elif mois_deb_test == 12:
				select_mois = 'Décembre 2020'
	#			st.write("Mois : Décembre 2020")
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