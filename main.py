import time
from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy

from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objs as go

# Экземпляр приложения
app = Flask(__name__)

# Настройки для подключения к базе данных
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Инициализация базы данных
db = SQLAlchemy(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    city = db.Column(db.String(80), nullable=False)

    def __repr__(self):
        return f'User({self.name}, {self.age}, {self.city})'


class Classification_results(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    dataset_name = db.Column(db.String(80), nullable=False)
    model_name = db.Column(db.String(80), nullable=False)
    accuracy = db.Column(db.Float, nullable=False)
    time = db.Column(db.Float, nullable=False)

    def __repr__(self):
        return f'Classification_results({self.dataset_name}, {self.model_name}, {self.accuracy}, {self.time})'


with app.app_context():
    db.create_all()


def fill_database():
    iris_data = load_iris()
    df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    df['species'] = iris_data.target

    # Разделение данных
    (X_train, X_test,
     y_train, y_test) = train_test_split(iris_data.data,
                                         iris_data.target, train_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "Support Vector Machines": SVC(kernel='rbf', gamma=0.1, C=1.0),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    }

    for name, model in models.items():
        start_t = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        end_t = time.time()
        new_result = Classification_results(dataset_name="iris", model_name=name,
                                            accuracy=round(accuracy_score(y_test, y_pred), 2),
                                            time=round(end_t - start_t, 6))
        db.session.add(new_result)

    wine_data = load_wine()
    df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
    df['types'] = wine_data.target

    # Разделение данных
    (X_train, X_test,
     y_train, y_test) = train_test_split(wine_data.data,
                                         wine_data.target, train_size=0.45, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=800),
        "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=3),
        "DecisionTreeClassifier": DecisionTreeClassifier(max_depth=5),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "GradientBoostingClassifier": GradientBoostingClassifier(n_estimators=40, learning_rate=0.9,
                                                                 max_depth=1, random_state=42)
    }

    for name, model in models.items():
        start_t = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        end_t = time.time()
        new_result = Classification_results(dataset_name="wine", model_name=name,
                                            accuracy=round(accuracy_score(y_test, y_pred), 2),
                                            time=round(end_t - start_t, 6))
        db.session.add(new_result)

    db.session.commit()


# Маршрут для главной страницы
@app.route('/', methods=['GET', 'POST'])
def home():
    global data
    greeting = ""
    plot_div = ""
    if request.method == 'POST':
        name = request.form.get('name')
        age = int(request.form.get('age'))
        city = request.form.get('city')

        greeting = f"Hello, {name} from {city}!"

        # Добавление данных в базу данных
        new_user = User(name=name, age=age, city=city)
        db.session.add(new_user)
        db.session.commit()

        users = User.query.all()
        data = [{'Name': user.name, 'Age': user.age, 'City': user.city} for user in users]

        fig = px.histogram(data, x='Age', title='Age Distribution', color_discrete_sequence=['#BD5353'])
        fig.update_layout({"bargap": 0.01})
        plot_div = pio.to_html(fig, full_html=False)

    return render_template('index.html', greeting=greeting, plot_div=plot_div)


@app.route('/iris', methods=['GET', 'POST'])
def iris():
    iris_data = load_iris()
    df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    df['species'] = iris_data.target

    plot_div = ""
    accuracy_results = ""

    scatter_matrix = px.scatter_matrix(
        df,
        dimensions=iris_data.feature_names,
        color='species',
        title='Iris Dataset Scatter Matrix',
        color_continuous_scale='magenta',
        labels={col: col.replace(" (cm)", "") for col in iris_data.feature_names}
    )
    plot_div = pio.to_html(scatter_matrix, full_html=False)

    if request.method == 'POST':
        # Разделение данных
        (X_train, X_test,
         y_train, y_test) = train_test_split(iris_data.data,
                                             iris_data.target, train_size=0.3, random_state=42)

        # Стандартизация
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Обучение моделей
        models = {
            "Logistic Regression": LogisticRegression(max_iter=200),
            "Support Vector Machines": SVC(kernel='rbf', gamma=0.1, C=1.0),
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        }

        accuracy_results = {}
        time_elapsed = {}
        for name, model in models.items():
            start_t = time.time()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            end_t = time.time()
            accuracy_results[name] = round(accuracy_score(y_test, y_pred), 2)
            time_elapsed[name] = round(end_t - start_t, 2)

        bar_chart = go.Figure([go.Bar(x=list(accuracy_results.keys()), y=list(accuracy_results.values()),
                                      marker=dict(color='#B0417A'))])
        bar_chart.update_layout(title="Model Accuracy Comparsion", xaxis_title="Model", yaxis_title="Accuracy")
        plot_div += pio.to_html(bar_chart, full_html=False)

    return render_template('iris.html', plot_div=plot_div, accuracy_results=accuracy_results)


@app.route('/wine', methods=['GET', 'POST'])
def wine():
    wine_data = load_wine()
    df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
    df['types'] = wine_data.target

    plot_div = ""
    plot_div1 = ""
    plot_div2 = ""
    accuracy_results = ""

    scatter_matrix = px.scatter_matrix(
        df,
        dimensions=['alcohol', 'ash','color_intensity', 'proline'],
        color='types',
        title='Wine Dataset Scatter Matrix',
        labels={col: col.replace(" (cm)", "") for col in wine_data.feature_names}
    )
    plot_div = pio.to_html(scatter_matrix, full_html=False)

    if request.method == 'POST':
        action = request.form['action']
        if action == 'update_plot':
            x_axis_feature = request.form['x_axis_feature']
            y_axis_feature = request.form['y_axis_feature']
            plot_type = request.form['plot_type']

            fig = px.scatter(df, x=x_axis_feature, y=y_axis_feature, color='types',
                             title='Interactive Wine Dataset Plot')
            if plot_type == 'histogram':
                fig = px.histogram(df, x=x_axis_feature, y=y_axis_feature, color='types',
                                   title='Histogram of Wine Dataset', nbins=8)
                fig.update_layout({"bargap": 0.01})
            elif plot_type == 'histogram with avg grouping':
                fig = px.histogram(df, x=x_axis_feature, y=y_axis_feature, color='types',
                                   title='Histogram of Wine Dataset with avg grouping',
                                   nbins=8, histfunc='avg')
                fig.update_layout({"bargap": 0.01})
            elif plot_type == 'box':
                fig = px.box(df, y=y_axis_feature, color='types',
                                   title='Box plot for Wine Dataset')

            # Re-render the plot
            plot_div1 = pio.to_html(fig, full_html=False)
            pass

        elif action == 'compare_models':

            (X_train, X_test,
            y_train, y_test) = train_test_split(wine_data.data,
                                             wine_data.target, train_size=0.45, random_state=42)

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            models = {
                "Logistic Regression": LogisticRegression(max_iter=800),
                "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=3),
                "DecisionTreeClassifier": DecisionTreeClassifier(max_depth=5),
                "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
                "GradientBoostingClassifier": GradientBoostingClassifier(n_estimators=40, learning_rate=0.9,
                                                                     max_depth=1, random_state=42)
            }

            accuracy_results = {}
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy_results[name] = round(accuracy_score(y_test, y_pred) * 100, 2)

            bar_chart = go.Figure([go.Bar(x=list(accuracy_results.keys()), y=list(accuracy_results.values()))])
            bar_chart.update_layout(title="Model Accuracy Comparsion", xaxis_title="Model", yaxis_title="Accuracy")
            plot_div2 = pio.to_html(bar_chart, full_html=False)
            pass

    return render_template('wine.html', tables=[df.to_html(classes='data', header=True)],
                           plot_div=plot_div, plot_div1=plot_div1, plot_div2=plot_div2,
                           accuracy_results=accuracy_results, wine_data=wine_data)


@app.route('/results', methods=['GET', 'POST'])
def results():
    global data
    plot_div = ""
    results = ""

    if request.method == 'POST':
        dataset_filter = request.form['dataset-filter']
        model_filter = request.form['model-filter']

        sort_by = request.args.get('sort_by', default='accuracy')
        sort_order = request.args.get('sort_order', default='desc')
        #Classification_results.query.delete()
        fill_database()

        if dataset_filter and model_filter:
            if sort_by == 'accuracy':
                if sort_order == 'desc':
                    results = Classification_results.query.filter_by(dataset_name=dataset_filter,
                    model_name=model_filter).order_by(Classification_results.accuracy.desc()).all()
                else:
                    results = Classification_results.query.filter_by(dataset_name=dataset_filter,
                    model_name=model_filter).order_by(Classification_results.accuracy.asc()).all()
            else:
                results = Classification_results.query.filter_by(dataset_name=dataset_filter,
                    model_name=model_filter).order_by(Classification_results.accuracy.desc()).all()
        elif dataset_filter:
            if sort_by == 'accuracy':
                if sort_order == 'desc':
                    results = Classification_results.query.filter_by(dataset_name=dataset_filter).order_by(
                        Classification_results.accuracy.desc()).all()
                else:
                    results = Classification_results.query.filter_by(dataset_name=dataset_filter).order_by(
                        Classification_results.accuracy.asc()).all()
            else:
                results = Classification_results.query.filter_by(dataset_name=dataset_filter).order_by(
                    Classification_results.accuracy.desc()).all()
        elif model_filter:
            if sort_by == 'accuracy':
                if sort_order == 'desc':
                    results = Classification_results.query.filter_by(model_name=model_filter).order_by(
                        Classification_results.accuracy.desc()).all()
                else:
                    results = Classification_results.query.filter_by(model_name=model_filter).order_by(
                        Classification_results.accuracy.asc()).all()
            else:
                results = Classification_results.query.filter_by(model_name=model_filter).order_by(
                    Classification_results.accuracy.desc()).all()
        else:
            results = Classification_results.query.order_by(Classification_results.accuracy.desc()).all()

    else:
        sort_by = request.args.get('sort_by', default='accuracy')
        sort_order = request.args.get('sort_order', default='desc')
        if sort_by == 'accuracy':
            if sort_order == 'desc':
                results = Classification_results.query.order_by(
                    Classification_results.accuracy.desc()).all()
            else:
                results = Classification_results.query.order_by(
                    Classification_results.accuracy.asc()).all()
        else:
            results = Classification_results.query.order_by(
                Classification_results.accuracy.desc()).all()

    datasets = db.session.query(Classification_results.dataset_name).distinct().all()
    models = db.session.query(Classification_results.model_name).distinct().all()
    return render_template('results.html', results=results, datasets=[dataset[0]
                                                                      for dataset in datasets],
                           models=[model[0] for model in models], sort_by=sort_by, sort_order=sort_order)


# Запуск
if __name__ == '__main__':
    app.run(debug=True)
