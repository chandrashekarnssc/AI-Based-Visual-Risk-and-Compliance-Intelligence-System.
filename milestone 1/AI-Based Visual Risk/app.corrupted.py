# app.py
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required,
logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import numpy as np
from sklearn.linear_model import LinearRegression
import os
from dotenv import load_dotenv
from collections import defaultdict
load_dotenv()
app = Flask(__name__)
app.config[&#39;SECRET_KEY&#39;] = os.getenv(&#39;SECRET_KEY&#39;, &#39;dev-key-change-in-production&#39;)
app.config[&#39;SQLALCHEMY_DATABASE_URI&#39;] = &#39;sqlite:///digital_twin.db&#39;
app.config[&#39;SQLALCHEMY_TRACK_MODIFICATIONS&#39;] = False
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = &#39;login&#39;
# ------------------------- Context Processor -------------------------
@app.context_processor
def utility_processor():
return dict(datetime=datetime)
# ------------------------- Gemini AI Setup -------------------------
try:
import google.generativeai as genai
GEMINI_AVAILABLE = True
except ImportError:
GEMINI_AVAILABLE = False

print(&quot;Warning: google-generativeai not installed. Install with: pip install google-
generativeai&quot;)
GEMINI_API_KEY = os.getenv(&#39;GEMINI_API_KEY&#39;)
if GEMINI_AVAILABLE and GEMINI_API_KEY:
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(&#39;gemini-2.5-flash&#39;)
print(&quot;Gemini initialized successfully.&quot;)
else:
gemini_model = None
if not GEMINI_API_KEY:
print(&quot;Warning: GEMINI_API_KEY not set. Assistant will use rule-based
responses.&quot;)
# ------------------------- Database Models -------------------------
class User(UserMixin, db.Model):
id = db.Column(db.Integer, primary_key=True)
username = db.Column(db.String(80), unique=True, nullable=False)
email = db.Column(db.String(120), unique=True, nullable=False)
password_hash = db.Column(db.String(200), nullable=False)
age = db.Column(db.Integer, default=25)
occupation = db.Column(db.String(100), default=&#39;Student&#39;)
monthly_income = db.Column(db.Float, default=50000)
monthly_expenses = db.Column(db.Float, default=35000)
savings = db.Column(db.Float, default=100000)
study_hours_per_week = db.Column(db.Float, default=20)
fitness_hours_per_week = db.Column(db.Float, default=5)
sleep_hours_per_day = db.Column(db.Float, default=7)
created_at = db.Column(db.DateTime, default=datetime.utcnow)
is_admin = db.Column(db.Boolean, default=False)
transactions = db.relationship(&#39;Transaction&#39;, backref=&#39;user&#39;, lazy=True)
study_logs = db.relationship(&#39;StudyLog&#39;, backref=&#39;user&#39;, lazy=True)
fitness_logs = db.relationship(&#39;FitnessLog&#39;, backref=&#39;user&#39;, lazy=True)
goals = db.relationship(&#39;Goal&#39;, backref=&#39;user&#39;, lazy=True)

def set_password(self, password):
self.password_hash = generate_password_hash(password)
def check_password(self, password):
return check_password_hash(self.password_hash, password)
class Transaction(db.Model):
id = db.Column(db.Integer, primary_key=True)
user_id = db.Column(db.Integer, db.ForeignKey(&#39;user.id&#39;), nullable=False)
date = db.Column(db.Date, nullable=False, default=datetime.utcnow)
category = db.Column(db.String(50))
amount = db.Column(db.Float, nullable=False)
type = db.Column(db.String(10))
class StudyLog(db.Model):
id = db.Column(db.Integer, primary_key=True)
user_id = db.Column(db.Integer, db.ForeignKey(&#39;user.id&#39;), nullable=False)
date = db.Column(db.Date, nullable=False, default=datetime.utcnow)
hours = db.Column(db.Float, nullable=False)
subject = db.Column(db.String(100))
productivity_score = db.Column(db.Integer)
class FitnessLog(db.Model):
id = db.Column(db.Integer, primary_key=True)
user_id = db.Column(db.Integer, db.ForeignKey(&#39;user.id&#39;), nullable=False)
date = db.Column(db.Date, nullable=False, default=datetime.utcnow)
activity = db.Column(db.String(50))
duration_min = db.Column(db.Integer, nullable=False)
calories_burned = db.Column(db.Integer)
class Goal(db.Model):
id = db.Column(db.Integer, primary_key=True)
user_id = db.Column(db.Integer, db.ForeignKey(&#39;user.id&#39;), nullable=False)
description = db.Column(db.String(200))
target_date = db.Column(db.Date)
achieved = db.Column(db.Boolean, default=False)

class ChatMessage(db.Model):
id = db.Column(db.Integer, primary_key=True)
user_id = db.Column(db.Integer, db.ForeignKey(&#39;user.id&#39;), nullable=False)
question = db.Column(db.Text, nullable=False)
response = db.Column(db.Text, nullable=False)
timestamp = db.Column(db.DateTime, default=datetime.utcnow)
user = db.relationship(&#39;User&#39;, backref=&#39;chat_messages&#39;)
class SimulationHistory(db.Model):
id = db.Column(db.Integer, primary_key=True)
user_id = db.Column(db.Integer, db.ForeignKey(&#39;user.id&#39;), nullable=False)
scenario = db.Column(db.String(50), nullable=False)
amount = db.Column(db.Float, nullable=False)
result_desc = db.Column(db.String(200))
result_impact = db.Column(db.String(200))
result_risk = db.Column(db.String(20))
result_rec = db.Column(db.String(200))
projected = db.Column(db.Float)
current = db.Column(db.Float)
timestamp = db.Column(db.DateTime, default=datetime.utcnow)
user = db.relationship(&#39;User&#39;, backref=&#39;simulations&#39;)
# ------------------------- Helper Functions -------------------------
def generate_forecast(data, periods=6):
if len(data) &lt; 2:
return [data[-1] if data else 0] * periods
X = np.array(range(len(data))).reshape(-1, 1)
y = np.array(data)
model = LinearRegression()
model.fit(X, y)
future_X = np.array(range(len(data), len(data)+periods)).reshape(-1, 1)
forecast = model.predict(future_X)
noise = np.random.normal(0, np.std(y)*0.1 if np.std(y) &gt; 0 else 1, periods)
return (forecast + noise).tolist()

def calculate_financial_metrics(user):
income = sum(t.amount for t in user.transactions if t.type == &#39;income&#39;)
expenses = sum(t.amount for t in user.transactions if t.type == &#39;expense&#39;)
net = income - expenses
rate = (net/income*100) if income &gt; 0 else 0
return {&#39;income&#39;: income, &#39;expenses&#39;: expenses, &#39;net&#39;: net, &#39;rate&#39;: rate}
def predict_gpa(user):
logs = user.study_logs
if not logs:
return {&#39;gpa&#39;: 3.0, &#39;efficiency&#39;: 70, &#39;recommendation&#39;: &#39;Log study sessions to get
predictions.&#39;}
total_hours = sum(l.hours for l in logs)
avg_prod = sum(l.productivity_score for l in logs) / len(logs)
gpa = 2.0 + (total_hours/100) + (avg_prod/100)
gpa = min(4.0, max(1.0, gpa))
if total_hours &lt; 20:
rec = &quot;Increase study hours.&quot;
elif avg_prod &lt; 70:
rec = &quot;Improve focus and quality.&quot;
else:
rec = &quot;Keep up the good work!&quot;
return {&#39;gpa&#39;: round(gpa, 2), &#39;efficiency&#39;: round(avg_prod, 1), &#39;recommendation&#39;: rec}
def simulate_scenario(user, action, amount):
scenarios = {
&#39;save&#39;: {
&#39;desc&#39;: f&#39;Save ₹{amount:,} monthly&#39;,
&#39;impact&#39;: f&#39;Your savings grow to ₹{user.savings + amount*12:,.0f} in 1 year.&#39;,
&#39;risk&#39;: &#39;Low&#39;,
&#39;rec&#39;: &#39;Great for long-term security&#39;
},
&#39;invest&#39;: {
&#39;desc&#39;: f&#39;Invest ₹{amount:,} monthly&#39;,

&#39;impact&#39;: f&#39;With 10% returns, investment grows to ₹{amount*12*1.1:,.0f} in 1
year.&#39;,
&#39;risk&#39;: &#39;Medium&#39;,
&#39;rec&#39;: &#39;Balanced approach for growth&#39;
},
&#39;spend&#39;: {
&#39;desc&#39;: f&#39;Spend ₹{amount:,} monthly extra&#39;,
&#39;impact&#39;: f&#39;Your savings reduce to ₹{user.savings - amount*12:,.0f} in 1 year.&#39;,
&#39;risk&#39;: &#39;High&#39;,
&#39;rec&#39;: &#39;Consider reducing discretionary spending&#39;
},
&#39;study_more&#39;: {
&#39;desc&#39;: f&#39;Study {amount} more hours per week&#39;,
&#39;impact&#39;: f&#39;Your GPA could increase by {amount*0.1:.2f} points.&#39;,
&#39;risk&#39;: &#39;Low&#39;,
&#39;rec&#39;: &#39;Consistent effort yields results&#39;
},
&#39;exercise&#39;: {
&#39;desc&#39;: f&#39;Exercise {amount} minutes daily&#39;,
&#39;impact&#39;: f&#39;You could burn {amount*30*30:,.0f} calories monthly.&#39;,
&#39;risk&#39;: &#39;Low&#39;,
&#39;rec&#39;: &#39;Great for overall health&#39;
}
}
return scenarios.get(action, scenarios[&#39;save&#39;])
# ------------------------- Default Admin Creator -------------------------
def create_default_admin():
&quot;&quot;&quot;Create a default admin user if none exists.&quot;&quot;&quot;
admin = User.query.filter_by(username=&#39;admin&#39;).first()
if not admin:
admin = User(
username=&#39;admin&#39;,
email=&#39;admin@digitaltwin.ai&#39;
)
admin.set_password(&#39;admin123&#39;)

admin.is_admin = True
db.session.add(admin)
db.session.commit()
print(&quot;✅ Default admin user created:&quot;)
print(&quot; Username: admin&quot;)
print(&quot; Password: admin123&quot;)
print(&quot; Please change your password after first login.&quot;)
else:
print(&quot;✅ Admin user already exists.&quot;)
# ------------------------- Routes -------------------------
@login_manager.user_loader
def load_user(user_id):
return User.query.get(int(user_id))
@app.route(&#39;/&#39;)
def index():
return render_template(&#39;index.html&#39;)
@app.route(&#39;/register&#39;, methods=[&#39;GET&#39;, &#39;POST&#39;])
def register():
if request.method == &#39;POST&#39;:
username = request.form.get(&#39;username&#39;, &#39;&#39;).strip()
email = request.form.get(&#39;email&#39;, &#39;&#39;).strip()
password = request.form.get(&#39;password&#39;, &#39;&#39;)
confirm_password = request.form.get(&#39;confirm_password&#39;, &#39;&#39;)
if not username or not email or not password:
flash(&#39;All fields are required.&#39;)
return redirect(url_for(&#39;register&#39;))
if password != confirm_password:
flash(&#39;Passwords do not match.&#39;)
return redirect(url_for(&#39;register&#39;))
if len(password) &lt; 8:
flash(&#39;Password must be at least 8 characters.&#39;)
return redirect(url_for(&#39;register&#39;))
if User.query.filter_by(username=username).first():

flash(&#39;Username already exists.&#39;)
return redirect(url_for(&#39;register&#39;))
if User.query.filter_by(email=email).first():
flash(&#39;Email already registered.&#39;)
return redirect(url_for(&#39;register&#39;))
user = User(username=username, email=email)
user.set_password(password)
# First user becomes admin – but now we already have a default admin, so this is
optional
# if User.query.count() == 0:
# user.is_admin = True
db.session.add(user)
db.session.commit()
flash(&#39;Registration successful! Please log in.&#39;)
return redirect(url_for(&#39;login&#39;))
return render_template(&#39;register.html&#39;)
@app.route(&#39;/login&#39;, methods=[&#39;GET&#39;, &#39;POST&#39;])
def login():
if request.method == &#39;POST&#39;:
username = request.form.get(&#39;username&#39;, &#39;&#39;).strip()
password = request.form.get(&#39;password&#39;, &#39;&#39;)
if not username or not password:
flash(&#39;Please enter both username and password.&#39;)
return render_template(&#39;login.html&#39;)
user = User.query.filter_by(username=username).first()
if user and user.check_password(password):
login_user(user)
return redirect(url_for(&#39;admin_dashboard&#39; if user.is_admin else
&#39;user_dashboard&#39;))
flash(&#39;Invalid username or password.&#39;)
return render_template(&#39;login.html&#39;)
@app.route(&#39;/logout&#39;)
@login_required
def logout():

logout_user()
return redirect(url_for(&#39;index&#39;))
# ------------------------- User Routes (unchanged) -------------------------
@app.route(&#39;/user/dashboard&#39;)
@login_required
def user_dashboard():

# ===== Existing metrics =====
metrics = calculate_financial_metrics(current_user)
gpa_info = predict_gpa(current_user)
total_fitness = sum(f.duration_min for f in current_user.fitness_logs)
# ===== Chart Data Preparation (as before) =====
monthly_income = defaultdict(float)
monthly_expenses = defaultdict(float)
for tx in current_user.transactions:
month_key = tx.date.strftime(&#39;%Y-%m&#39;)
if tx.type == &#39;income&#39;:
monthly_income[month_key] += tx.amount
else:
monthly_expenses[month_key] += tx.amount
all_months = sorted(set(monthly_income.keys()) | set(monthly_expenses.keys()))
months = all_months[-6:]
income_data = [monthly_income.get(m, 0) for m in months]
expense_data = [monthly_expenses.get(m, 0) for m in months]
study_dates = defaultdict(float)
for log in current_user.study_logs:
study_dates[log.date.strftime(&#39;%Y-%m-%d&#39;)] += log.hours
study_days = sorted(study_dates.keys())[-14:]
study_hours = [study_dates.get(d, 0) for d in study_days]
fitness_dates = defaultdict(float)

for log in current_user.fitness_logs:
fitness_dates[log.date.strftime(&#39;%Y-%m-%d&#39;)] += log.duration_min
fitness_days = sorted(fitness_dates.keys())[-14:]
fitness_minutes = [fitness_dates.get(d, 0) for d in fitness_days]
goal_progress = []
for goal in current_user.goals:
if not goal.achieved and goal.target_date:
days_remaining = (goal.target_date - datetime.now().date()).days
if days_remaining &lt; 0:
progress = 100
else:
progress = min(100, max(0, 100 - (days_remaining / 30 * 100)))
goal_progress.append({&#39;description&#39;: goal.description, &#39;progress&#39;:
round(progress, 1)})
# ===== PREDICTIVE ANALYTICS =====
# Savings forecast (same as before)
monthly_net = defaultdict(float)
for tx in current_user.transactions:
month_key = tx.date.strftime(&#39;%Y-%m&#39;)
if tx.type == &#39;income&#39;:
monthly_net[month_key] += tx.amount
else:
monthly_net[month_key] -= tx.amount
sorted_months = sorted(monthly_net.keys())
net_values = [monthly_net[m] for m in sorted_months[-6:]]
if len(net_values) &gt;= 3:
forecast_net = generate_forecast(net_values, 6)
savings_forecast = []
cum = current_user.savings
for val in forecast_net:
cum += val
savings_forecast.append(cum)
else:
savings_forecast = [current_user.savings + i * 5000 for i in range(6)]

# GPA forecast
study_hours_by_month = defaultdict(float)
for log in current_user.study_logs:
month_key = log.date.strftime(&#39;%Y-%m&#39;)
study_hours_by_month[month_key] += log.hours
study_hours_list = [study_hours_by_month[m] for m in
sorted(study_hours_by_month.keys())[-4:]]
if len(study_hours_list) &gt;= 3:
forecast_hours = generate_forecast(study_hours_list, 3)
gpa_forecast = []
avg_prod = sum(l.productivity_score for l in current_user.study_logs) /
len(current_user.study_logs) if current_user.study_logs else 70
for hours in forecast_hours:
gpa = 2.0 + (hours / 50) + (avg_prod / 200)
gpa_forecast.append(round(min(4.0, max(1.0, gpa)), 2))
else:
gpa_forecast = [gpa_info[&#39;gpa&#39;] + i*0.05 for i in range(3)]
# Fitness projection – use the same logic as before
fitness_by_day = defaultdict(float)
for log in current_user.fitness_logs:
fitness_by_day[log.date.strftime(&#39;%Y-%m-%d&#39;)] += log.duration_min
fitness_days_list = list(fitness_by_day.values())[-14:]
if len(fitness_days_list) &gt;= 5:
forecast_fitness = generate_forecast(fitness_days_list, 30)
projected_fitness = sum(forecast_fitness)
else:
projected_fitness = total_fitness * 2
# ===== EXTRA METRICS =====
total_study_hours = sum(log.hours for log in current_user.study_logs) or 0
# Study streak (consecutive days with at least some study in last 7 days)
study_streak = 0
if current_user.study_logs:

study_days_set = set(log.date for log in current_user.study_logs)
today = datetime.now().date()
for i in range(7):
if today - timedelta(days=i) in study_days_set:
study_streak += 1
else:
break
# Fitness streak (consecutive days with &gt;= 10 min activity)
fitness_streak = 0
if current_user.fitness_logs:
today = datetime.now().date()
for i in range(7):
day_total = sum(l.duration_min for l in current_user.fitness_logs if l.date ==
(today - timedelta(days=i)))
if day_total &gt;= 10:
fitness_streak += 1
else:
break
# Subject breakdown
subject_totals = defaultdict(float)
for log in current_user.study_logs:
if log.subject:
subject_totals[log.subject] += log.hours
subjects = list(subject_totals.keys())
subject_hours = list(subject_totals.values())
# Activity breakdown
activity_totals = defaultdict(float)
for log in current_user.fitness_logs:
activity_totals[log.activity] += log.duration_min
activities = list(activity_totals.keys())
activity_minutes = list(activity_totals.values())
# Life Score

life_score = (
(metrics[&#39;rate&#39;] / 30) * 20 +
(gpa_info[&#39;gpa&#39;] / 4) * 30 +
(min(total_fitness, 300) / 300) * 25 +
(len([g for g in current_user.goals if not g.achieved]) / max(1,
len(current_user.goals))) * 25
)
life_score = round(life_score)
# ===== RENDER =====
return render_template(&#39;user/dashboard.html&#39;,
user=current_user,
metrics=metrics,
gpa=gpa_info,
total_fitness=total_fitness,
months=months,
income_data=income_data,
expense_data=expense_data,
study_days=study_days,
study_hours=study_hours,
fitness_days=fitness_days,
fitness_minutes=fitness_minutes,
goal_progress=goal_progress,
savings_forecast=savings_forecast,
gpa_forecast=gpa_forecast,
projected_fitness=projected_fitness,
total_study_hours=total_study_hours,
study_streak=study_streak,
fitness_streak=fitness_streak,
subjects=subjects,
subject_hours=subject_hours,
activities=activities,
activity_minutes=activity_minutes,
life_score=life_score)
@app.route(&#39;/user/financial&#39;, methods=[&#39;GET&#39;, &#39;POST&#39;])

@login_required
def user_financial():

if request.method == &#39;POST&#39;:
date = datetime.strptime(request.form[&#39;date&#39;], &#39;%Y-%m-%d&#39;)
category = request.form[&#39;category&#39;]
amount = float(request.form[&#39;amount&#39;])
trans_type = request.form[&#39;type&#39;]
tx = Transaction(user_id=current_user.id, date=date, category=category,
amount=amount, type=trans_type)
db.session.add(tx)
db.session.commit()
flash(&#39;Transaction added.&#39;)
return redirect(url_for(&#39;user_financial&#39;))
transactions = current_user.transactions
expense_data = [t.amount for t in transactions if t.type == &#39;expense&#39;]
forecast = generate_forecast(expense_data, 6) if expense_data else []
# Metrics
metrics = calculate_financial_metrics(current_user)
# Category breakdown for pie chart (only expenses)
category_totals = {}
for tx in transactions:
if tx.type == &#39;expense&#39;:
category_totals[tx.category] = category_totals.get(tx.category, 0) + tx.amount
categories = list(category_totals.keys())
category_amounts = list(category_totals.values())
# For the recent transactions list, we&#39;ll limit to last 10
recent_transactions = transactions[-10:][::-1] # newest first
return render_template(&#39;user/financial.html&#39;,
user=current_user,

transactions=recent_transactions,
forecast=forecast,
metrics=metrics,
categories=categories,
category_amounts=category_amounts)

@app.route(&#39;/edit_transaction&#39;, methods=[&#39;POST&#39;])
@login_required
def edit_transaction():
&quot;&quot;&quot;Update an existing transaction (called from the modal).&quot;&quot;&quot;
tx_id = request.form.get(&#39;id&#39;)
if not tx_id:
flash(&#39;Transaction ID missing.&#39;, &#39;danger&#39;)
return redirect(url_for(&#39;user_financial&#39;))
# Retrieve the transaction and ensure it belongs to the current user
tx = Transaction.query.filter_by(id=tx_id, user_id=current_user.id).first()
if not tx:
flash(&#39;Transaction not found or you do not have permission.&#39;, &#39;danger&#39;)
return redirect(url_for(&#39;user_financial&#39;))
# Update fields
try:
tx.date = datetime.strptime(request.form[&#39;date&#39;], &#39;%Y-%m-%d&#39;).date()
tx.category = request.form[&#39;category&#39;]
tx.amount = float(request.form[&#39;amount&#39;])
tx.type = request.form[&#39;type&#39;]
db.session.commit()
flash(&#39;Transaction updated successfully!&#39;, &#39;success&#39;)
except Exception as e:
db.session.rollback()
flash(f&#39;Error updating transaction: {str(e)}&#39;, &#39;danger&#39;)
return redirect(url_for(&#39;user_financial&#39;))

@app.route(&#39;/delete_transaction/&lt;int:id&gt;&#39;, methods=[&#39;POST&#39;])
@login_required
def delete_transaction(id):
&quot;&quot;&quot;Delete a transaction by ID (only if it belongs to the current user).&quot;&quot;&quot;
tx = Transaction.query.filter_by(id=id, user_id=current_user.id).first()
if not tx:
flash(&#39;Transaction not found or you do not have permission.&#39;, &#39;danger&#39;)
return redirect(url_for(&#39;user_financial&#39;))
try:
db.session.delete(tx)
db.session.commit()
flash(&#39;Transaction deleted.&#39;, &#39;info&#39;)
except Exception as e:
db.session.rollback()
flash(f&#39;Error deleting transaction: {str(e)}&#39;, &#39;danger&#39;)
return redirect(url_for(&#39;user_financial&#39;))
@app.route(&#39;/user/study&#39;, methods=[&#39;GET&#39;, &#39;POST&#39;])
@login_required
def user_study():
if request.method == &#39;POST&#39;:
date = datetime.strptime(request.form[&#39;date&#39;], &#39;%Y-%m-%d&#39;)
hours = float(request.form[&#39;hours&#39;])
subject = request.form[&#39;subject&#39;]
productivity = int(request.form[&#39;productivity&#39;])
log = StudyLog(user_id=current_user.id, date=date, hours=hours, subject=subject,
productivity_score=productivity)
db.session.add(log)
db.session.commit()
flash(&#39;Study session logged.&#39;, &#39;success&#39;)
return redirect(url_for(&#39;user_study&#39;))
logs = current_user.study_logs

gpa_info = predict_gpa(current_user) # your existing helper
# --- Data for charts ---
study_by_day = defaultdict(float)
for log in logs:
study_by_day[log.date.strftime(&#39;%Y-%m-%d&#39;)] += log.hours
study_dates = sorted(study_by_day.keys())[-14:]
study_hours = [study_by_day.get(d, 0) for d in study_dates]
subject_totals = defaultdict(float)
for log in logs:
if log.subject:
subject_totals[log.subject] += log.hours
subjects = list(subject_totals.keys())
subject_hours = list(subject_totals.values())
prod_by_day = defaultdict(list)
for log in logs:
prod_by_day[log.date.strftime(&#39;%Y-%m-%d&#39;)].append(log.productivity_score)
prod_dates = sorted(prod_by_day.keys())[-14:]
prod_avg = [sum(prod_by_day[d])/len(prod_by_day[d]) if d in prod_by_day else 0
for d in prod_dates]
total_hours = sum(log.hours for log in logs)
avg_productivity = sum(log.productivity_score for log in logs) / len(logs) if logs else 0
# Study streak
study_streak = 0
if logs:
study_days_set = sorted(set(log.date for log in logs), reverse=True)
today = datetime.now().date()
for i in range(7):
check_date = today - timedelta(days=i)
if check_date in study_days_set:
study_streak += 1
else:

break
recent_logs = logs[-10:][::-1]
return render_template(&#39;user/study.html&#39;,
user=current_user,
logs=recent_logs,
gpa=gpa_info,
study_dates=study_dates,
study_hours=study_hours,
subjects=subjects,
subject_hours=subject_hours,
prod_dates=prod_dates,
prod_avg=prod_avg,
total_hours=total_hours,
avg_productivity=avg_productivity,
study_streak=study_streak)

@app.route(&#39;/edit_study_log&#39;, methods=[&#39;POST&#39;])
@login_required
def edit_study_log():
log_id = request.form.get(&#39;id&#39;)
if not log_id:
flash(&#39;Log ID missing.&#39;, &#39;danger&#39;)
return redirect(url_for(&#39;user_study&#39;))
log = StudyLog.query.filter_by(id=log_id, user_id=current_user.id).first()
if not log:
flash(&#39;Log not found or you do not have permission.&#39;, &#39;danger&#39;)
return redirect(url_for(&#39;user_study&#39;))
try:
log.date = datetime.strptime(request.form[&#39;date&#39;], &#39;%Y-%m-%d&#39;).date()
log.hours = float(request.form[&#39;hours&#39;])
log.subject = request.form.get(&#39;subject&#39;)

log.productivity_score = int(request.form[&#39;productivity&#39;])
db.session.commit()
flash(&#39;Study log updated successfully!&#39;, &#39;success&#39;)
except Exception as e:
db.session.rollback()
flash(f&#39;Error updating log: {str(e)}&#39;, &#39;danger&#39;)
return redirect(url_for(&#39;user_study&#39;))

@app.route(&#39;/delete_study_log/&lt;int:id&gt;&#39;, methods=[&#39;POST&#39;])
@login_required
def delete_study_log(id):
log = StudyLog.query.filter_by(id=id, user_id=current_user.id).first()
if not log:
flash(&#39;Log not found or you do not have permission.&#39;, &#39;danger&#39;)
return redirect(url_for(&#39;user_study&#39;))
try:
db.session.delete(log)
db.session.commit()
flash(&#39;Study log deleted.&#39;, &#39;info&#39;)
except Exception as e:
db.session.rollback()
flash(f&#39;Error deleting log: {str(e)}&#39;, &#39;danger&#39;)
return redirect(url_for(&#39;user_study&#39;))
@app.route(&#39;/user/fitness&#39;, methods=[&#39;GET&#39;, &#39;POST&#39;])
@login_required
def user_fitness():

if request.method == &#39;POST&#39;:
date = datetime.strptime(request.form[&#39;date&#39;], &#39;%Y-%m-%d&#39;)
activity = request.form[&#39;activity&#39;]

duration = int(request.form[&#39;duration&#39;])
calories = int(request.form[&#39;calories&#39;])
log = FitnessLog(user_id=current_user.id, date=date, activity=activity,
duration_min=duration, calories_burned=calories)
db.session.add(log)
db.session.commit()
flash(&#39;Fitness activity logged.&#39;)
return redirect(url_for(&#39;user_fitness&#39;))
logs = current_user.fitness_logs
# --- Basic metrics ---
total_min = sum(l.duration_min for l in logs)
total_cal = sum(l.calories_burned for l in logs)
avg_min_per_day = total_min / max(1,
(datetime.now().date() - (logs[0].date if logs else
datetime.now().date())).days)
# --- Weekly progress (current week: Monday–Sunday) ---
today = datetime.now().date()
start_of_week = today - timedelta(days=today.weekday()) # Monday
weekly_logs = [l for l in logs if l.date &gt;= start_of_week]
weekly_min = sum(l.duration_min for l in weekly_logs)
week_goal = 150 # minutes per week
week_progress = min(100, (weekly_min / week_goal) * 100)
# --- Daily trend (last 14 days) ---
from collections import defaultdict
daily_totals = defaultdict(int)
for log in logs:
daily_totals[log.date.strftime(&#39;%Y-%m-%d&#39;)] += log.duration_min
# Get last 14 days (including today)
dates = [(today - timedelta(days=i)).strftime(&#39;%Y-%m-%d&#39;) for i in range(13, -1, -1)]
daily_data = [daily_totals.get(d, 0) for d in dates]
# --- Activity breakdown ---

activity_totals = defaultdict(int)
for log in logs:
activity_totals[log.activity] += log.duration_min
activities = list(activity_totals.keys())
activity_minutes = list(activity_totals.values())
# --- Streak: consecutive days with at least 10 min of activity ---
streak = 0
if logs:
study_days_set = sorted(set(log.date for log in logs), reverse=True)
for i in range(30): # max 30 days streak
check_date = today - timedelta(days=i)
# Check if there is any activity on this day (&gt;=10 min)
day_total = sum(l.duration_min for l in logs if l.date == check_date)
if day_total &gt;= 10:
streak += 1
else:
break
# --- Recent logs (last 10, newest first) ---
recent_logs = logs[-10:][::-1]
return render_template(&#39;user/fitness.html&#39;,
user=current_user,
logs=recent_logs,
total_min=total_min,
total_cal=total_cal,
avg_min_per_day=round(avg_min_per_day, 1),
weekly_min=weekly_min,
week_progress=round(week_progress, 1),
dates=dates,
daily_data=daily_data,
activities=activities,
activity_minutes=activity_minutes,
streak=streak)
@app.route(&#39;/fitness/edit/&lt;int:log_id&gt;&#39;, methods=[&#39;POST&#39;])

@login_required
def edit_fitness(log_id):
log = FitnessLog.query.get_or_404(log_id)
if log.user_id != current_user.id:
flash(&#39;You are not authorised to edit this activity.&#39;)
return redirect(url_for(&#39;user_fitness&#39;))
date = datetime.strptime(request.form[&#39;date&#39;], &#39;%Y-%m-%d&#39;)
activity = request.form[&#39;activity&#39;]
duration = int(request.form[&#39;duration&#39;])
calories = int(request.form[&#39;calories&#39;])
log.date = date
log.activity = activity
log.duration_min = duration
log.calories_burned = calories
db.session.commit()
flash(&#39;Activity updated successfully! ✅&#39;)
return redirect(url_for(&#39;user_fitness&#39;))

@app.route(&#39;/fitness/delete/&lt;int:log_id&gt;&#39;, methods=[&#39;POST&#39;])
@login_required
def delete_fitness(log_id):
log = FitnessLog.query.get_or_404(log_id)
if log.user_id != current_user.id:
flash(&#39;You are not authorised to delete this activity.&#39;)
return redirect(url_for(&#39;user_fitness&#39;))
db.session.delete(log)
db.session.commit()

flash(&#39;Activity deleted successfully!

��
️&#39;) return redirect(url_for(&#39;user_fitness&#39;))@app.route(&#39;/user/simulation&#39;, methods=[&#39;GET&#39;, &#39;POST&#39;])@login_requireddef user_simulation(): result = None current_values = { &#39;savings&#39;: current_user.savings, &#39;gpa&#39;: predict_gpa(current_user)[&#39;gpa&#39;], &#39;fitness&#39;: sum(f.duration_min for f in current_user.fitness_logs) } if request.method == &#39;POST&#39;: scenario = request.form[&#39;scenario&#39;] amount = float(request.form[&#39;amount&#39;]) result = simulate_scenario(current_user, scenario, amount) result[&#39;scenario&#39;] = scenario # Compute projections (full code below) if scenario == &#39;save&#39;: result[&#39;projected&#39;] = current_user.savings + amount * 12 result[&#39;current&#39;] = current_user.savings result[&#39;impact&#39;] = f&quot;Your savings grow to ₹{result[&#39;projected&#39;]:,.0f} in 1 year.&quot; elif scenario == &#39;invest&#39;: result[&#39;projected&#39;] = amount * 12 * 1.1 result[&#39;current&#39;] = 0 result[&#39;impact&#39;] = f&quot;Your investment could grow to ₹{result[&#39;projected&#39;]:,.0f} in 1 year.&quot; elif scenario == &#39;spend&#39;: result[&#39;projected&#39;] = current_user.savings - amount * 12 result[&#39;current&#39;] = current_user.savings result[&#39;impact&#39;] = f&quot;Your savings reduce to ₹{result[&#39;projected&#39;]:,.0f} in 1 year.&quot; elif scenario == &#39;study_more&#39;: current_gpa = predict_gpa(current_user)[&#39;gpa&#39;] efficiency = predict_gpa(current_user)[&#39;efficiency&#39;] result[&#39;projected&#39;] = round(2.0 + (amount / 50) + (efficiency / 200), 2) result[&#39;current&#39;] = current_gpa result[&#39;impact&#39;] = f&quot;Your GPA could improve from {current_gpa:.2f} to {result[&#39;projected&#39;]:.2f}.&quot; elif scenario == &#39;exercise&#39;: result[&#39;projected&#39;] = amount * 30 * 30 result[&#39;current&#39;] = sum(f.duration_min for f in current_user.fitness_logs) result[&#39;impact&#39;] = f&quot;You could burn {result[&#39;projected&#39;]:,.0f} calories per month.&quot; # ---- Save to history ---- history = SimulationHistory( user_id=current_user.id, scenario=scenario, amount=amount, result_desc=result[&#39;desc&#39;], result_impact=result[&#39;impact&#39;], result_risk=result[&#39;risk&#39;], result_rec=result[&#39;rec&#39;], projected=result.get(&#39;projected&#39;, 0), current=result.get(&#39;current&#39;, 0) ) db.session.add(history) db.session.commit() # ---- Render directly with result ---- history_entries = SimulationHistory.query.filter_by(user_id=current_user.id).order_by(SimulationHistory.timestamp.desc()).all() return render_template(&#39;user/simulation.html&#39;, user=current_user, result=result, current_values=current_values, history=history_entries) # ---- GET: fetch history ---- history_entries = SimulationHistory.query.filter_by(user_id=current_user.id).order_by(SimulationHistory.timestamp.desc()).all() return render_template(&#39;user/simulation.html&#39;, user=current_user, result=result, current_values=current_values, history=history_entries)@app.route(&#39;/user/simulation/clear&#39;, methods=[&#39;POST&#39;])@login_requireddef clear_simulation_history(): SimulationHistory.query.filter_by(user_id=current_user.id).delete() db.session.commit() flash(&#39;Simulation history cleared.&#39;) return redirect(url_for(&#39;user_simulation&#39;))@app.route(&#39;/user/assistant&#39;, methods=[&#39;GET&#39;, &#39;POST&#39;])@login_requireddef user_assistant(): # ----- Clear session for new chat ----- if request.args.get(&#39;new&#39;): session.pop(&#39;last_query&#39;, None) session.pop(&#39;last_response&#39;, None) session[&#39;hide_messages&#39;] = True return redirect(url_for(&#39;user_assistant&#39;)) # ----- Load a specific conversation from history ----- load_id = request.args.get(&#39;load&#39;) if load_id: msg = ChatMessage.query.filter_by(id=load_id, user_id=current_user.id).first() if msg: session[&#39;last_query&#39;] = msg.question session[&#39;last_response&#39;] = msg.response session[&#39;hide_messages&#39;] = False else: flash(&#39;Message not found.&#39;) return redirect(url_for(&#39;user_assistant&#39;)) # ----- Handle POST (new message) ----- if request.method == &#39;POST&#39;: query = request.form.get(&#39;query&#39;, &#39;&#39;).strip() if not query: flash(&#39;Please enter a question.&#39;) return redirect(url_for(&#39;user_assistant&#39;)) # --- Build user context for prompt --- metrics = calculate_financial_metrics(current_user) gpa_info = predict_gpa(current_user) total_fitness = sum(f.duration_min for f in current_user.fitness_logs) study_total = sum(log.hours for log in current_user.study_logs) goals = [g.description for g in current_user.goals if not g.achieved] user_context = f&quot;&quot;&quot; User Profile: - Age: {current_user.age} - Occupation: {current_user.occupation} - Monthly Income: ₹{current_user.monthly_income:,.0f} - Monthly Expenses: ₹{current_user.monthly_expenses:,.0f} - Savings: ₹{current_user.savings:,.0f} - Savings Rate: {metrics[&#39;rate&#39;]:.1f}% - Study Hours/Week: {current_user.study_hours_per_week:.1f} - Fitness Hours/Week: {current_user.fitness_hours_per_week:.1f} - Sleep Hours/Day: {current_user.sleep_hours_per_day:.1f} - Predicted GPA: {gpa_info[&#39;gpa&#39;]}/4.0 - Total Study Hours Logged: {study_total:.1f} - Total Fitness Minutes Logged: {total_fitness} - Active Goals: {&#39;, &#39;.join(goals) if goals else &#39;None&#39;} &quot;&quot;&quot; # --- Only Gemini – no fallback --- if not gemini_model or not GEMINI_API_KEY: flash(&#39;⚠

️ Gemini is not configured. Please set GEMINI_API_KEY.&#39;) return redirect(url_for(&#39;user_assistant&#39;)) try: prompt = f&quot;&quot;&quot; You are a helpful personal decision assistant. Answer the user&#39;s question based on their data. {user_context} User question: {query} Provide actionable, concise, and encouraging advice. &quot;&quot;&quot; gemini_response = gemini_model.generate_content(prompt) response = gemini_response.text.strip() except Exception as e: app.logger.error(f&quot;Gemini error: {e}&quot;) flash(&#39;⚠
️ Gemini service is currently unavailable. Please try again later.&#39;) return redirect(url_for(&#39;user_assistant&#39;)) # Save to database chat_msg = ChatMessage(user_id=current_user.id, question=query, response=response) db.session.add(chat_msg) db.session.commit() session[&#39;last_query&#39;] = query session[&#39;last_response&#39;] = response session[&#39;hide_messages&#39;] = False return redirect(url_for(&#39;user_assistant&#39;)) # ----- GET: show chat ----- last_query = session.get(&#39;last_query&#39;) last_response = session.get(&#39;last_response&#39;) hide_messages = session.get(&#39;hide_messages&#39;, False) all_messages = ChatMessage.query.filter_by(user_id=current_user.id).order_by(ChatMessage.timestamp.desc()).all()