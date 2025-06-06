from flask import Flask, render_template, request, jsonify
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import csv
import re
from datetime import datetime
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep
from scipy.stats import norm
import os

app = Flask(__name__)
df = None

def clean_number(val):
    if isinstance(val, str):
        val = re.sub(r'[^\d\.\-]', '', val)
    try:
        return float(val)
    except:
        return np.nan

def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))
        return f'{pct:.1f}%\n({val})'
    return my_autopct

def generate_charts(df, selected_metric, chart_type):
    if df is None:
        return None

    labels = df['Group'].values
    values = df[selected_metric].apply(clean_number).values

    if np.isnan(values).any():
        return None

    if chart_type == "pie":
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(values, labels=labels, autopct=make_autopct(values),
               startangle=90, colors=plt.cm.viridis(np.linspace(0.2, 0.8, len(values))),
               textprops={'color': 'white'})
        ax.set_title(selected_metric, fontsize=14, weight='bold', color='white')
        fig.patch.set_facecolor('none')
    elif chart_type == "bar":
        fig, ax = plt.subplots(figsize=(6, 6))
        bars = ax.bar(labels, values, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(values))))
        ax.set_title(selected_metric, fontsize=14, weight='bold', color='white')
        ax.tick_params(colors='white')
        ax.set_facecolor('#121212')
        fig.patch.set_facecolor('#121212')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.2f}',
                    ha='center', va='bottom', color='white')
    elif chart_type == "table":
        table_html = df[['Group', selected_metric]].to_html(classes='styled-table', index=False)
        return table_html

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    global df
    try:
        data = request.form['audience_data']
        start_date_str = request.form.get('start_date')
        selected_metric = request.form.get('metric', 'Visitors')
        chart_type = request.form.get('chart_type', 'pie')
        significance_level = float(request.form.get('significance_level', '0.05'))
        ci_method = request.form.get('ci_method', 'wald')
        test_power = float(request.form.get('test_power', '0.8'))

        rows = list(csv.reader(data.strip().splitlines(), delimiter='\t'))
        if len(rows) < 2:
            raise ValueError("Invalid data")

        df = pd.DataFrame(rows[1:], columns=rows[0])
        chart = generate_charts(df, selected_metric, chart_type)

        visitors = df['Visitors'].apply(clean_number).values
        orders = df['Orders'].apply(clean_number).values
        groups = df['Group'].values
        aov = df['Average Order Value'].apply(clean_number).values

        if len(groups) < 2:
            raise ValueError("At least two groups are required")

        conv_a, conv_b = orders[0], orders[1]
        n_a, n_b = visitors[0], visitors[1]

        cr_a = conv_a / n_a if n_a > 0 else 0
        cr_b = conv_b / n_b if n_b > 0 else 0

        pval, zstat = proportions_ztest([conv_a, conv_b], [n_a, n_b], alternative='two-sided')
        pval = round(abs(pval), 6)
        zstat = round(zstat, 2)
        is_significant = pval < significance_level

        ci_low, ci_upp = confint_proportions_2indep(conv_a, n_a, conv_b, n_b, method=ci_method)
        ci_low = round(ci_low * 100, 2)
        ci_upp = round(ci_upp * 100, 2)

        lift = ((cr_b - cr_a) / cr_a) * 100 if cr_a > 0 else 0
        lift = round(lift, 2)

        se = np.sqrt((cr_a * (1 - cr_a)) / n_a + (cr_b * (1 - cr_b)) / n_b) * 100
        se = round(se, 2)

        increased_orders = orders[1] - orders[0]
        aov_diff = aov[1] - aov[0]
        real_additional_value = increased_orders * aov_diff

        days_to_significance = 'N/A'
        if start_date_str:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
            days_elapsed = (datetime.today() - start_date).days
            if days_elapsed > 0 and cr_b != cr_a:
                total_visitors = n_a + n_b
                daily_visitors = total_visitors / days_elapsed

                z_crit = abs(norm.ppf(significance_level / 2)) + abs(norm.ppf(test_power))
                min_required_visitors = (z_crit ** 2) * (cr_a * (1 - cr_a) + cr_b * (1 - cr_b)) / (cr_b - cr_a) ** 2
                visitors_remaining = max(min_required_visitors - total_visitors, 0)
                estimated_days_left = visitors_remaining / daily_visitors if daily_visitors > 0 else 'N/A'
                days_to_significance = round(estimated_days_left, 1)

        stats_result = {
            "group_a": groups[0],
            "group_b": groups[1],
            "cr_a": round(cr_a * 100, 2),
            "cr_b": round(cr_b * 100, 2),
            "pval": pval,
            "zstat": zstat,
            "significant": is_significant,
            "confidence_interval": f"{ci_low}% to {ci_upp}%",
            "real_additional_value": round(real_additional_value, 2),
            "lift": lift,
            "se": se,
            "days_to_significance": days_to_significance
        }

        return render_template('index.html', chart=chart, stats=stats_result,
                               selected_metric=selected_metric, chart_type=chart_type,
                               significance_level=significance_level)

    except Exception as e:
        return render_template('index.html', error=str(e))

@app.route('/update_chart', methods=['POST'])
def update_chart():
    try:
        selected_metric = request.form.get('metric', 'Visitors')
        chart_type = request.form.get('chart_type', 'pie')
        chart = generate_charts(df, selected_metric, chart_type)

    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
