{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Eficiência Energética em Edificações \n",
    "## Uma aplicação de Machine Learning\n",
    "\n",
    "Este trabalho tem o objetivo de aplicar o conhecimento recém-adquirido em “Machine Learning”, com a utilização de técnicas de aprendizado supervisionado e métricas de avaliação.  \n",
    "\n",
    "O tema do estudo é a previsão de eficiência energética em edificações.   A referência é o estudo [“Accurate quantitative estimation of energy performance of residential buildings using statistical machine learning tools”](http://people.maths.ox.ac.uk/tsanas/Preprints/ENB2012.pdf), de Athanasios Tsanas e Angeliki Xifara (2012), em que foram realizados experimentos usando 12 edifícios de diferentes formas, para avaliação da eficiência energética em edificações como função dos parâmetros de construção.  \n",
    "A base de dados foi obtida no website “Machine Learning Repository” da Universidade da Califórnia, Irvine (UCI): [“Energy efficiency Data Set”](https://archive.ics.uci.edu/ml/datasets/Energy+efficiency), \n",
    "\n",
    "Trata-se de um modelo de aprendizado supervisionado, pois existe uma variável (_target_) que se deseja prever – duas, na verdade: demanda de energia para aquecimento (_‘heating load’_) e demanda de energia para resfriamento (_‘cooling load’_). As variáveis _target_ são explicadas por características da edificação (_‘features’_), tais como área de paredes, área envidraçada, entre outros atributos.   \n",
    "Por meio da seleção das técnicas mais apropriadas para o tipo de problema colocado, busca-se comparar os modelos e encontrar o que apresenta melhor ajuste para a previsão de demanda de energia.  \n",
    "\n",
    "O projeto possui uma fase de análise exploratória de dados e uma fase de modelagem. Para a modelagem, foram comparados pelas métricas de avaliação r2 e RMSE o Modelo de K-vizinhos mais próximos (KNN), Modelo de Regressão Linear, Modelo de Regressão Linear Ridge, Modelo de Regressão Linear Lasso e Modelo Ensemble (GradientBoostingRegressor), sendo este último considerado o melhor modelo de acordo com as métricas de avaliação.  \n",
    "\n",
    "O projeto foi desenvolvido em um [Jupyter Notebook](./Scripts/building_energy_efficiency.ipynb)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
