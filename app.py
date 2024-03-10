import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import pymc3 as pm
import math
from sklearn.preprocessing import StandardScaler
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QComboBox,
    QPushButton,
    QVBoxLayout,
    QMessageBox,
    QFileDialog,
    QDialog,
    QTableWidget,
    QTableWidgetItem,
    QHBoxLayout,
    QDateTimeEdit,
    QScrollArea,
)

import networkx as nx
import sqlite3
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from io import StringIO
from causalnex.structure.notears import from_pandas
import matplotlib.pyplot as plt

from causalnex.structure import StructureModel
import arviz as az
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeRegressor

from causalnex.structure import StructureModel


class HistoryDialog(QDialog):
    def __init__(self, history):
        super().__init__()
        self.setWindowTitle("Hystory")
        self.layout = QVBoxLayout()

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Data", "hours", "Country", "Year"])

        for i, (date, time, country, year) in enumerate(history):
            self.table.insertRow(i)
            self.table.setItem(i, 0, QTableWidgetItem(date))
            self.table.setItem(i, 1, QTableWidgetItem(time))
            self.table.setItem(i, 2, QTableWidgetItem(country))
            self.table.setItem(i, 3, QTableWidgetItem(str(year)))

        self.layout.addWidget(self.table)
        self.setLayout(self.layout)


class ReportDialog(QDialog):
    def __init__(
        self,
        country,
        year,
        dnn_prediction,
        y_pred_bayesian,
        filtered_data,
        rmse_nn,
        mape_nn,
        theil_nn,
        rmse_bayesian,
        mape_bayesian,
        theil_bayesian,
        alpha,
        beta,
        sigma,
    ):
        super().__init__()
        self.setWindowTitle("Report")
        self.setFixedSize(900, 700)  # Define tamanho fixo da janela

        self.layout = QVBoxLayout()

        temperatura = filtered_data["temperature"].iloc[0]
        co2_emissions = filtered_data["co2_emissions"].iloc[0]
        divulgacao = filtered_data["tcfd_disclosure"].iloc[0]
        governanca = filtered_data["tcfd_governance"].iloc[0]
        estrategia = filtered_data["tcfd_strategy"].iloc[0]
        grisco = filtered_data["tcfd_risk_management"].iloc[0]
        metrica = filtered_data["tcfd_metrics_targets"].iloc[0]
        precipitacao = filtered_data["precipitation"].iloc[0]
        performing_loans_ratio = filtered_data["performing_loans_ratio"].iloc[0]
        capital_ratio = filtered_data["capital_ratio"].iloc[0]
        product_gap = filtered_data["product_gap"].iloc[0]
        report_text = (
            f"<html><body>"
            f'<p style="font-family: Arial; font-size:16px; font-weight:bold"> Report on Climate Change and TCFD Recommendations</p>'
            f'<p style="font-family: Arial; font-size:14px;"> Country: {country}</p>'
            f'<p style="font-family: Arial; font-size:14px;"> Year: {year}</p>'
            f'<p style="font-family: Arial; font-size:16px; font-weight:bold">Credit Risk</p>'
            f'<p style="font-family: Arial; font-size:14px;">The value of Performing Loans Ratio for the selected country and year is {performing_loans_ratio:.4f} % .</p>'
            f'<p style="font-family: Arial; font-size:14px;">The value of Capital Ratio for the selected country and year is {capital_ratio:.4f} %.</p>'
            f'<p style="font-family: Arial; font-size:14px;">The value of Product Gap for the selected country and year is {product_gap:.4f} %.</p>'
            f'<p style="font-family: Arial; font-size:14px;">The predicted value of credit risk using the recurrent neural network model is {dnn_prediction:.3f}.</p>'
            f'<p style="font-family: Arial; font-size:14px;">The predicted value of RMSE the recurrent neural network model is {rmse_nn:.3f}.</p>'
            f'<p style="font-family: Arial; font-size:14px;">The predicted value of MAPA the recurrent neural network model is {mape_nn:.3f}.</p>'
            f'<p style="font-family: Arial; font-size:14px;">The predicted value of Theil the recurrent neural network model is {theil_nn:.3f}.</p>'
            f'<p style="font-family: Arial; font-size:14px;">The predicted value of  the recurrent neural network model is {rmse_nn:.3f}.</p>'
            f'<p style="font-family: Arial; font-size:14px;">The predicted value of MAPA the recurrent neural network model is {mape_nn:.3f}.</p>'
            f'<p style="font-family: Arial; font-size:14px;">The predicted value of Theil the recurrent neural network model is {theil_nn:.3f}.</p>'
            f'<p style="font-family: Arial; font-size:14px;">The estimated value of credit risk using the Bayesian model is {y_pred_bayesian:.3f}.</p>'
            f'<p style="font-family: Arial; font-size:14px;">The predicted value of RMSE the using the Bayesian model is is {rmse_bayesian:.3f}.</p>'
            f'<p style="font-family: Arial; font-size:14px;">The predicted value of MAPA the using the Bayesian model is is {mape_bayesian:.3f}.</p>'
            f'<p style="font-family: Arial; font-size:14px;">The predicted value of Theil the using the Bayesian model is is {theil_bayesian:.3f}.</p>'
            f'<p style="font-family: Arial; font-size:14px;">The predicted value of Sigma the using the Bayesian model is is {sigma:.3f}.</p>'
            f'<p style="font-family: Arial; font-size:14px;">The predicted value of Alpha the using the Bayesian model is is {alpha:.3f}.</p>'
            f'<p style="font-family: Arial; font-size:14px;">The predicted value of beta the using the Bayesian model is is {beta:.3f}.</p>'
            f'<p style="font-family: Arial; font-size:16px; font-weight:bold"> Climate Change</p>'
            f'<p style="font-family: Arial; font-size:14px;">Climate change refers to alterations in Earth is climate primarily caused by the increase in greenhouse gas emissions, such as carbon dioxide (CO2), resulting from human activities. Climate change has significant impacts on temperature, precipitation, sea level, biodiversity, health, agriculture, energy, economy, and security.</p>'
            f"<ul>"
            f'<li><span style="font-family: Arial; font-size:14px">Temperature: {temperatura:.2f} degrees Celsius</span></li>'
            f'<li><span style="font-family: Arial; font-size:14px">Precipitation: {precipitacao:.2f} millimeters</span></li>'
            f'<li><span style="font-family: Arial; font-size:14px">CO2 Emissions: {co2_emissions:.2f} tons per capita</span></li>'
            f"</ul>"
            f'<p style="font-family: Arial; font-size:16px; font-weight:bold"> TCFD Recommendations</p>'
            f'<p style="font-family: Arial; font-size:14px">The Task Force on Climate-related Financial Disclosures (TCFD) recommendations are a set of voluntary guidelines for organizations to disclose climate-related financial information. TCFD recommendations cover four main areas: disclosure, governance, risk management strategy, and metrics and targets.</p>'
            f"<ul>"
            f'<li><span style="font-family: Arial; font-size:14px">Disclosure: {divulgacao}</span></li>'
            f'<li><span style="font-family: Arial; font-size:14px">Governance: {governanca}</span></li>'
            f'<li><span style="font-family: Arial; font-size:14px">Strategy: {estrategia}</span></li>'
            f'<li><span style="font-family: Arial; font-size:14px">Risk Management: {grisco}</span></li>'
            f'<li><span style="font-family: Arial; font-size:14px">Metrics and Targets: {metrica}</span></li>'
            f"</ul>"
            f'<p style="font-family: Arial; font-size:16px; font-weight:bold"> Conclusion</p>'
            f"<ul>"
            f'<li><span style="font-family: Arial; font-size:14px">Based on the results from recurrent neural network and structural causal models, we can conclude that:</span></li>'
            f'<li><span style="font-family: Arial; font-size:14px">The credit risk of Eurozone banks is influenced by climate change and TCFD recommendations.</span></li>'
            f'<li><span style="font-family: Arial; font-size:14px">Increasing temperature, precipitation, and CO2 emissions increase the credit risk of Eurozone banks.</span></li>'
            f'<li><span style="font-family: Arial; font-size:14px">Implementing TCFD recommendations reduces the credit risk of Eurozone banks.</span></li>'
            f'<li><span style="font-family: Arial; font-size:14px">Recurrent neural network and structural causal models can assist Eurozone banks in improving their compliance with TCFD recommendations by providing accurate predictions and causal inferences on credit risk concerning climate change.</span></li>'
            f"</ul><br><br>"
            f"</body></html>"
        )

        # Configurando a label com o texto formatado
        self.label = QLabel(report_text)
        self.label.setAlignment(Qt.AlignJustify)  # Justifica o texto
        self.label.setWordWrap(True)  # Habilita a quebra de linha
        self.layout.addWidget(self.label)

        # Criando a área de rolagem apenas na vertical
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.label)
        scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarAlwaysOn
        )  # Adiciona barra de rolagem apenas na vertical

        self.layout.addWidget(
            scroll_area
        )  # Adiciona a área de rolagem ao layout principal

        # Definindo o tamanho da fonte
        font = QFont("Arial", 12)
        self.label.setFont(font)

        self.setLayout(self.layout)

        # Configurando o estilo da janela
        self.setStyleSheet("background-color: white; padding: 5px;")


class RiskCalculatorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Generete Report")
        self.setFixedSize(500, 400)  # Definindo o tamanho da tela

        self.layout = QVBoxLayout()

        # Carregar o arquivo DataFrame
        self.data_file_label = QLabel("Select file:")
        self.layout.addWidget(self.data_file_label)
        self.load_data_button = QPushButton("open faile")
        self.load_data_button.clicked.connect(self.load_data)
        self.layout.addWidget(self.load_data_button)

        # ComboBox para seleção de país
        self.country_label = QLabel("Select Country:")
        self.layout.addWidget(self.country_label)
        self.country_combo = QComboBox()
        self.layout.addWidget(self.country_combo)

        # ComboBox para seleção de ano
        self.year_label = QLabel("Select Year:")
        self.layout.addWidget(self.year_label)
        self.year_combo = QComboBox()
        self.layout.addWidget(self.year_combo)

        # Botões
        self.buttons_layout = QHBoxLayout()

        self.calculate_button = QPushButton("Generete Report")
        self.calculate_button.clicked.connect(self.show_report)
        self.buttons_layout.addWidget(self.calculate_button)

        self.history_button = QPushButton("Vew History")
        self.history_button.clicked.connect(self.show_history)
        self.buttons_layout.addWidget(self.history_button)

        self.show_causal_model_button = QPushButton("train Model")
        self.show_causal_model_button.clicked.connect(self.show_causal_model)
        self.buttons_layout.addWidget(self.show_causal_model_button)

        self.layout.addLayout(self.buttons_layout)

        self.setLayout(self.layout)
        self.data = None
        self.db_connection = None

    def load_data(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load file", "", "file  CSV (*.csv)"
        )
        if file_path:
            self.data = pd.read_csv(file_path)
            self.populate_comboboxes()
            self.data_file_label.setText(f"file: {file_path}")

    def populate_comboboxes(self):
        if self.data is not None:
            countries = self.data["country"].unique()
            self.country_combo.addItems(countries)

            years = self.data["year"].unique()
            self.year_combo.addItems(map(str, years))

    def show_report(self):
        if self.data is None:
            QMessageBox.warning(
                self, "Aviso", "Por favor, carregue um arquivo de dados primeiro."
            )
            return

        country = self.country_combo.currentText()
        year = int(self.year_combo.currentText())

        # Filtrar os dados para o país e ano selecionados
        filtered_data = self.data[
            (self.data["country"] == country) & (self.data["year"] == year)
        ]
        w1 = filtered_data["performing_loans_ratio"]
        w2 = filtered_data["product_gap"]
        w3 = filtered_data["capital_ratio"]
        sum1 = w1 * math.sqrt(255) + w2
        sum2 = w2 * math.sqrt(255) + w1
        sum3 = w3 * math.sqrt(255) + w3

        volatilidadeComposta = w1 + w2 + w3
        # Verificar se há dados suficientes
        if len(filtered_data) < 2:
            print(
                "There are not enough data to split between training and test sets. Using all the data for training."
            )
            X = filtered_data[["temperature", "precipitation", "co2_emissions"]]
            y = volatilidadeComposta
            X_scaled = StandardScaler().fit_transform(X)
            X_train, y_train = X_scaled, y
            X_test, y_test = X_scaled, y
        else:
            X = filtered_data[["temperature", "precipitation", "co2_emissions"]]
            y = volatilidadeComposta

            # Normalizar os dados
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Dividir os dados em conjuntos de treinamento e teste
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

        # Construir o modelo de rede neural profunda com camadas LSTM
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Reshape(
                    (X_train.shape[1], 1), input_shape=(X_train.shape[1],)
                ),
                tf.keras.layers.LSTM(64),
                tf.keras.layers.Dense(1),
            ]
        )

        # Compilar o modelo
        model.compile(loss="mse", optimizer="adam")

        # Treinar o modelo
        model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

        # Avaliar o modelo
        dnn_prediction = model.predict(X_test)
        rmse_nn = np.sqrt(mean_squared_error(y_test, dnn_prediction))
        mape_nn = (
            np.mean(
                np.abs(
                    (y_test.to_numpy() - dnn_prediction.flatten()) / y_test.to_numpy()
                )
            )
            * 100
        )
        theil_nn = np.sqrt(
            mean_squared_error(y_test, dnn_prediction)
            / (
                mean_squared_error(y_test, dnn_prediction)
                + mean_squared_error(y_test, y_test)
            )
        )

        # Construir o modelo bayesiano

        with pm.Model() as bayesian_model:
            alpha = pm.Normal("alpha", mu=0, sigma=10)
            beta = pm.Normal("beta", mu=0, sigma=10, shape=X_train.shape[1])
            sigma = pm.HalfNormal("sigma", sigma=1)

            mu = alpha + pm.math.dot(X_train, beta)
            y_pred_bayesian = pm.Normal(
                "y_pred_bayesian", mu=mu, sigma=sigma, observed=y_train
            )

            trace = pm.sample(1000, tune=1000, cores=1)

        # Avaliar o modelo bayesiano
        y_pred_trace = pm.sample_posterior_predictive(
            trace, samples=500, model=bayesian_model
        )
        y_pred_bayesian = np.mean(y_pred_trace["y_pred_bayesian"], axis=0)
        rmse_bayesian = np.sqrt(mean_squared_error(y_test, y_pred_bayesian))
        mape_bayesian = (
            np.mean(
                np.abs(
                    (y_test.to_numpy() - y_pred_bayesian.flatten()) / y_test.to_numpy()
                )
            )
            * 100
        )
        theil_bayesian = np.sqrt(
            mean_squared_error(y_test, y_pred_bayesian)
            / (
                mean_squared_error(y_test, y_pred_bayesian)
                + mean_squared_error(y_test, y_test)
            )
        )

        # Construir e treinar o modelo de regressão linear
        linear_model = LinearRegression()
        linear_model.fit(X_train, y_train)

        # Fazer previsões com o modelo treinado
        linear_predictions = linear_model.predict(X_test)

        # Calcular métricas de erro para regressão linear
        rmse_linear = np.sqrt(mean_squared_error(y_test, linear_predictions))
        mape_linear = np.mean(np.abs((y_test - linear_predictions) / y_test)) * 100
        theil_linear = np.sqrt(
            mean_squared_error(y_test, linear_predictions)
            / (
                mean_squared_error(y_test, linear_predictions)
                + mean_squared_error(y_test, y_test)
            )
        )
        # Exiba os resultados
        print("Resultado do modelo Decision Tree:")
        print("MAE:", rmse_linear)
        print("R2:", mape_linear)
        print("theail:", theil_linear)
        print("\nResultado do modelo bayesiano:")
        print(
            "Valor de risco de crédito previsto pelo modelo bayesiano:",
            np.mean(y_pred_bayesian),
        )
        print("RMSE:", rmse_bayesian)
        print("MAPE:", mape_bayesian)
        print("Índice de Theil:", theil_bayesian)
        dnn_value = np.mean(dnn_prediction)
        bayesian_value = np.mean(y_pred_bayesian)

        # Save predictions to CSV
        predictions_df = pd.DataFrame(
            {
                "country": [country],
                "year": [year],
                "dnn_value": [dnn_value],
                "bayesian_value": [bayesian_value],
                "rmse_nn": [rmse_nn],
                "mape_nn": [mape_nn],
                "theil_nn": [theil_nn],
                "rmse_bayesian": [rmse_bayesian],
                "mape_bayesian": [mape_bayesian],
                "theil_bayesian": [theil_bayesian],
                "performing_loans_ratio": [sum1],
                "product_gap": [sum2],
                "capital_ratio": [sum3],
            }
        )
        predictions_df.to_csv("predictions.csv", index=False)

        print("Resultado do modelo de rede neural profunda:")
        print(
            "Predicted credit risk value by the structural causal model.:",
            np.mean(dnn_prediction),
        )
        print("MAPE:", mape_nn)
        print("Índice de Theil:", theil_nn)
        print("\nResultado do modelo bayesiano:")
        print(
            "Valor de risco de crédito previsto pelo modelo bayesiano:",
            np.mean(y_pred_bayesian),
        )
        print("RMSE:", rmse_bayesian)
        print("MAPE:", mape_bayesian)
        print("Índice de Theil:", theil_bayesian)
        dnn_value = np.mean(dnn_prediction)
        bayesian_value = np.mean(y_pred_bayesian)

        # Save predictions to CSV
        predictions_df = pd.DataFrame(
            {
                "country": [country],
                "year": [year],
                "dnn_value": [dnn_value],
                "bayesian_value": [bayesian_value],
                "rmse_nn": [rmse_nn],
                "mape_nn": [mape_nn],
                "theil_nn": [theil_nn],
                "rmse_bayesian": [rmse_bayesian],
                "mape_bayesian": [mape_bayesian],
                "theil_bayesian": [theil_bayesian],
                "performing_loans_ratio": [sum1],
                "product_gap": [sum2],
                "capital_ratio": [sum3],
            }
        )
        salpha = trace["alpha"].mean()
        sbeta = trace["beta"].mean()
        ssigma = trace["sigma"].mean()
        predictions_df.to_csv("predictions.csv", index=False)

        # Store data in SQLite database

        # Mostrar o relatório com os valores calculados
        dialog = ReportDialog(
            country,
            year,
            dnn_value,
            bayesian_value,
            filtered_data,
            rmse_nn,
            mape_nn,
            theil_nn,
            rmse_bayesian,
            mape_bayesian,
            theil_bayesian,
            salpha,
            sbeta,
            ssigma,
        )
        dialog.exec_()

    def show_history(self):
        if self.db_connection is None:
            self.db_connection = sqlite3.connect("risk_predictions.db")

        cursor = self.db_connection.cursor()
        cursor.execute("SELECT * FROM history")
        history = cursor.fetchall()

        dialog = HistoryDialog(history)
        dialog.exec_()

    def show_causal_model(self):
        # Implemente a lógica para exibir o modelo causal
        if self.data is None:
            QMessageBox.warning(
                self, "Aviso", "Por favor, carregue um arquivo de dados primeiro."
            )
            return
        sm = StructureModel()
        country = self.country_combo.currentText()
        year = int(self.year_combo.currentText())

        # Filtrar os dados para o país e ano selecionados
        filtered_data = self.data[
            (self.data["country"] == country) & (self.data["year"] == year)
        ]
        dfp = pd.get_dummies(filtered_data, columns=["country"])
        sm = from_pandas(dfp)

        # Adicionar as arestas que representam as relações causais entre as variáveis
        sm.add_edges_from(
            [
                ("temperature", "credit_risk"),
                ("precipitation", "credit_risk"),
                ("co2_emissions", "credit_risk"),
            ]
        )

        # Remover as arestas que têm um valor de p inferior a 0.05
        for edge in list(sm.edges()):
            if sm[edge[0]][edge[1]]["weight"] < 0.05:
                sm.remove_edge(edge[0], edge[1])

        # Remover os nós que não têm nenhuma aresta
        for node in sm.nodes():
            if sm.degree(node) == 0:
                sm.remove_node(node)

        plt.figure(figsize=(8, 6))
        nx.draw(sm, with_labels=True)
        plt.title("Modelo Estrutural Causal")
        plt.show()
        pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RiskCalculatorApp()
    window.show()
    sys.exit(app.exec_())
