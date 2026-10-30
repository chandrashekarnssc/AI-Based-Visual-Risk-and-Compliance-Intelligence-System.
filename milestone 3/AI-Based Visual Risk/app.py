"""Visual Risk Flask application."""

from collections import defaultdict
from datetime import date, datetime, timedelta
import html
import math
import os

from dotenv import load_dotenv
from flask import Flask, flash, redirect, render_template, render_template_string, request, session, url_for
from flask_login import LoginManager, UserMixin, current_user, login_required, login_user, logout_user
from flask_sqlalchemy import SQLAlchemy
from jinja2 import FileSystemLoader
from werkzeug.security import check_password_hash, generate_password_hash


class UnescapingTemplateLoader(FileSystemLoader):
    """Loads the existing templates after converting HTML entities back to text."""

    def get_source(self, environment, template):
        source, filename, uptodate = super().get_source(environment, template)
        return html.unescape(source), filename, uptodate


load_dotenv()
app = Flask(__name__, template_folder="templets")
app.jinja_loader = UnescapingTemplateLoader(os.path.join(app.root_path, "templets"))
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-key-change-in-production")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///digital_twin.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

login_manager = LoginManager(app)
login_manager.login_view = "login"


@app.context_processor
def inject_datetime():
    return {"datetime": datetime}


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    age = db.Column(db.Integer, default=25)
    occupation = db.Column(db.String(100), default="Student")
    monthly_income = db.Column(db.Float, default=50000)
    monthly_expenses = db.Column(db.Float, default=35000)
    savings = db.Column(db.Float, default=100000)
    study_hours_per_week = db.Column(db.Float, default=20)
    fitness_hours_per_week = db.Column(db.Float, default=5)
    sleep_hours_per_day = db.Column(db.Float, default=7)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_admin = db.Column(db.Boolean, default=False)
    transactions = db.relationship("Transaction", backref="user", lazy=True)
    study_logs = db.relationship("StudyLog", backref="user", lazy=True)
    fitness_logs = db.relationship("FitnessLog", backref="user", lazy=True)
    goals = db.relationship("Goal", backref="user", lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    date = db.Column(db.Date, nullable=False, default=date.today)
    category = db.Column(db.String(50), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    type = db.Column(db.String(10), nullable=False)


class StudyLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    date = db.Column(db.Date, nullable=False, default=date.today)
    hours = db.Column(db.Float, nullable=False)
    subject = db.Column(db.String(100), nullable=False)
    productivity_score = db.Column(db.Integer, nullable=False)


class FitnessLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    date = db.Column(db.Date, nullable=False, default=date.today)
    activity = db.Column(db.String(50), nullable=False)
    duration_min = db.Column(db.Integer, nullable=False)
    calories_burned = db.Column(db.Integer, default=0)


class Goal(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    description = db.Column(db.String(200), nullable=False)
    target_date = db.Column(db.Date)
    achieved = db.Column(db.Boolean, default=False)


class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    question = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user = db.relationship("User", backref="chat_messages")


class SimulationHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    scenario = db.Column(db.String(50), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    result_desc = db.Column(db.String(200), nullable=False)
    result_impact = db.Column(db.String(200), nullable=False)
    result_risk = db.Column(db.String(20), nullable=False)
    result_rec = db.Column(db.String(200), nullable=False)
    projected = db.Column(db.Float, nullable=False)
    current = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user = db.relationship("User", backref="simulations")


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


def forecast_values(values, periods):
    """Project a simple linear trend without requiring optional ML packages."""
    if not values:
        return [0] * periods
    if len(values) == 1:
        return [float(values[0])] * periods

    x_mean = (len(values) - 1) / 2
    y_mean = sum(values) / len(values)
    denominator = sum((x - x_mean) ** 2 for x in range(len(values)))
    slope = sum(
        (x - x_mean) * (value - y_mean) for x, value in enumerate(values)
    ) / denominator
    return [
        max(0, y_mean + slope * (len(values) + step - 1 - x_mean))
        for step in range(periods)
    ]


def user_gpa(user):
    if not user.study_logs:
        return 3.0
    total_hours = sum(log.hours for log in user.study_logs)
    average_productivity = sum(
        log.productivity_score for log in user.study_logs
    ) / len(user.study_logs)
    return round(min(4.0, max(1.0, 2.0 + total_hours / 100 + average_productivity / 100)), 2)


def simulate_scenario(user, scenario, amount):
    savings = user.savings or 0
    if scenario == "save":
        projected = savings + amount * 12
        return {
            "desc": f"Save ₹{amount:,.0f} monthly",
            "impact": f"Your savings could grow to ₹{projected:,.0f} in one year.",
            "risk": "Low",
            "rec": "Keep the amount sustainable and build an emergency fund.",
            "current": savings,
            "projected": projected,
        }
    if scenario == "invest":
        projected = savings + amount * 12 * 1.1
        return {
            "desc": f"Invest ₹{amount:,.0f} monthly",
            "impact": (
                f"At an illustrative 10% uplift, your balance could reach "
                f"₹{projected:,.0f} in one year; returns are not guaranteed."
            ),
            "risk": "Medium",
            "rec": "Consider your risk tolerance and keep emergency savings accessible.",
            "current": savings,
            "projected": projected,
        }
    if scenario == "spend":
        projected = savings - amount * 12
        return {
            "desc": f"Spend ₹{amount:,.0f} more monthly",
            "impact": f"Your savings could change to ₹{projected:,.0f} in one year.",
            "risk": "High",
            "rec": "Check your monthly budget before committing to extra spending.",
            "current": savings,
            "projected": projected,
        }
    if scenario == "study_more":
        current = user_gpa(user)
        projected = round(min(4.0, current + amount * 0.02), 2)
        return {
            "desc": f"Study {amount:g} additional hours per week",
            "impact": f"Your estimated GPA could move from {current:.2f} to {projected:.2f}.",
            "risk": "Low",
            "rec": "Use focused sessions and review your progress regularly.",
            "current": current,
            "projected": projected,
        }

    current_calories = sum(log.calories_burned or 0 for log in user.fitness_logs)
    projected = current_calories + amount * 5 * 30
    return {
        "desc": f"Exercise {amount:g} additional minutes daily",
        "impact": (
            f"That adds about {amount * 5 * 30:,.0f} estimated calories over 30 days "
            "(actual results vary)."
        ),
        "risk": "Low",
        "rec": "Increase gradually and choose an activity that feels sustainable.",
        "current": current_calories,
        "projected": projected,
    }


def local_assistant_response(query, user):
    """Provide useful personalized guidance when an external AI key is unavailable."""
    question = query.lower()
    parts = []
    if any(word in question for word in ("save", "money", "budget", "income", "expense", "invest")):
        monthly_gap = (user.monthly_income or 0) - (user.monthly_expenses or 0)
        parts.append(
            f"Your profile shows monthly income of ₹{user.monthly_income or 0:,.0f}, "
            f"expenses of ₹{user.monthly_expenses or 0:,.0f}, and savings of "
            f"₹{user.savings or 0:,.0f}. The estimated monthly gap is ₹{monthly_gap:,.0f}. "
            "Review recurring costs and set an automatic savings target you can maintain."
        )
    if any(word in question for word in ("study", "gpa", "exam", "school", "college")):
        study_hours = sum(log.hours for log in user.study_logs)
        parts.append(
            f"Your estimated GPA is {user_gpa(user):.2f}/4.0, based on "
            f"{study_hours:g} logged study hours. Try short focused sessions, "
            "then log productivity so you can see which routine works."
        )
    if any(word in question for word in ("fitness", "exercise", "health", "workout")):
        exercise_minutes = sum(log.duration_min for log in user.fitness_logs)
        parts.append(
            f"You have logged {exercise_minutes} fitness minutes. Build a regular, "
            "manageable routine and increase activity gradually."
        )
    if any(word in question for word in ("goal", "plan", "motivat", "advice")) or not parts:
        active_goals = [goal.description for goal in user.goals if not goal.achieved]
        goal_text = ", ".join(active_goals[:3]) if active_goals else "no active goals yet"
        parts.append(
            f"You currently have {goal_text}. Choose one specific next step for this "
            "week and track it in your dashboard."
        )
    return "\n\n".join(parts)


SIMPLE_PAGE = """
<!doctype html><title>{{ title }} · Visual Risk</title>
<main style="font-family:Arial;max-width:760px;margin:3rem auto;padding:0 1rem">
  <p><a href="{{ url_for('index') }}">← Home</a></p>
  {% for message in get_flashed_messages() %}<p style="background:#fff3cd;padding:.7rem">{{ message }}</p>{% endfor %}
  <h1>{{ title }}</h1>{{ body|safe }}
</main>
"""


def simple_page(title, body):
    return render_template_string(SIMPLE_PAGE, title=title, body=body)


@app.route("/")
def index():
    # This deliberately retains the existing index.html -> base.html relationship.
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    # return render_template("register.html")
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        if not username or not email or not password:
            flash('All fields are required.')
            return redirect(url_for('register'))
        if password != confirm_password:
            flash('Passwords do not match.')
            return redirect(url_for('register'))
        if len(password) < 8:
            flash('Password must be at least 8 characters.')
            return redirect(url_for('register'))
        if User.query.filter_by(username=username).first():
            flash('Username already exists.')
            return redirect(url_for('register'))
        if User.query.filter_by(email=email).first():
            flash('Email already registered.')
            return redirect(url_for('register'))
        user = User(username=username, email=email)
        user.set_password(password)
        # First user becomes admin – but now we already have a default admin, so this is optional
        # if User.query.count() == 0:
        #     user.is_admin = True
        db.session.add(user)
        db.session.commit()
        flash('Registration successful! Please log in.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("user_dashboard"))
    if request.method == "POST":
        user = User.query.filter_by(username=request.form.get("username", "").strip()).first()
        if user and user.check_password(request.form.get("password", "")):
            login_user(user)
            return redirect(url_for("user_dashboard"))
        flash("Invalid username or password.")
    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("index"))


@app.route("/user/dashboard")
@login_required
def user_dashboard():
    transactions = sorted(current_user.transactions, key=lambda item: item.date)
    study_logs = sorted(current_user.study_logs, key=lambda item: item.date)
    fitness_logs = sorted(current_user.fitness_logs, key=lambda item: item.date)
    goals = current_user.goals

    income = sum(tx.amount for tx in transactions if tx.type == "income")
    expenses = sum(tx.amount for tx in transactions if tx.type == "expense")
    net = income - expenses
    metrics = {
        "income": income,
        "expenses": expenses,
        "net": net,
        "rate": net / income * 100 if income else 0,
    }
    total_study_hours = sum(log.hours for log in study_logs)
    total_fitness = sum(log.duration_min for log in fitness_logs)
    average_productivity = (
        sum(log.productivity_score or 0 for log in study_logs) / len(study_logs)
        if study_logs
        else 70
    )
    gpa_value = (
        min(4.0, max(1.0, 2.0 + total_study_hours / 100 + average_productivity / 100))
        if study_logs
        else 3.0
    )
    gpa = {"gpa": round(gpa_value, 2), "efficiency": round(average_productivity, 1)}

    monthly_income = defaultdict(float)
    monthly_expenses = defaultdict(float)
    monthly_net = defaultdict(float)
    for tx in transactions:
        month = tx.date.strftime("%Y-%m")
        if tx.type == "income":
            monthly_income[month] += tx.amount
            monthly_net[month] += tx.amount
        else:
            monthly_expenses[month] += tx.amount
            monthly_net[month] -= tx.amount
    months = sorted(set(monthly_income) | set(monthly_expenses))[-6:]

    study_by_day = defaultdict(float)
    study_by_month = defaultdict(float)
    subject_totals = defaultdict(float)
    for log in study_logs:
        study_by_day[log.date.strftime("%Y-%m-%d")] += log.hours
        study_by_month[log.date.strftime("%Y-%m")] += log.hours
        if log.subject:
            subject_totals[log.subject] += log.hours
    study_days = sorted(study_by_day)[-14:]

    fitness_by_day = defaultdict(float)
    activity_totals = defaultdict(float)
    for log in fitness_logs:
        fitness_by_day[log.date.strftime("%Y-%m-%d")] += log.duration_min
        if log.activity:
            activity_totals[log.activity] += log.duration_min
    fitness_days = sorted(fitness_by_day)[-14:]

    monthly_net_values = [monthly_net[month] for month in sorted(monthly_net)[-6:]]
    savings_forecast = forecast_values(monthly_net_values, 6)
    balance = current_user.savings or 0
    savings_forecast = [
        (balance := max(0, balance + value)) for value in savings_forecast
    ]
    monthly_study = [study_by_month[month] for month in sorted(study_by_month)[-4:]]
    if len(monthly_study) >= 3:
        gpa_forecast = [
            round(min(4.0, max(1.0, 2.0 + hours / 50 + average_productivity / 200)), 2)
            for hours in forecast_values(monthly_study, 3)
        ]
    else:
        gpa_forecast = [
            round(min(4.0, max(1.0, gpa_value + (index + 1) * 0.05)), 2)
            for index in range(3)
        ]
    projected_fitness = sum(
        forecast_values(list(fitness_by_day.values())[-14:], 30)
    ) if len(fitness_by_day) >= 5 else total_fitness * 2

    today = date.today()
    study_dates = {log.date for log in study_logs}
    study_streak = 0
    streak_date = today
    while streak_date in study_dates:
        study_streak += 1
        streak_date -= timedelta(days=1)
    fitness_streak = 0
    for offset in range(7):
        active_minutes = sum(
            log.duration_min
            for log in fitness_logs
            if log.date == today - timedelta(days=offset)
        )
        if active_minutes >= 10:
            fitness_streak += 1
        else:
            break

    life_score = round(min(100, max(
        0,
        (metrics["rate"] / 30) * 20
        + (gpa["gpa"] / 4) * 30
        + (min(total_fitness, 300) / 300) * 25
        + (sum(not goal.achieved for goal in goals) / max(1, len(goals))) * 25,
    )))

    return render_template(
        "user/dashboard.html",
        user=current_user,
        metrics=metrics,
        gpa=gpa,
        total_fitness=total_fitness,
        total_study_hours=total_study_hours,
        study_streak=study_streak,
        fitness_streak=fitness_streak,
        life_score=life_score,
        months=months,
        income_data=[monthly_income[month] for month in months],
        expense_data=[monthly_expenses[month] for month in months],
        study_days=study_days,
        study_hours=[study_by_day[day] for day in study_days],
        fitness_days=fitness_days,
        fitness_minutes=[fitness_by_day[day] for day in fitness_days],
        subjects=list(subject_totals),
        subject_hours=list(subject_totals.values()),
        activities=list(activity_totals),
        activity_minutes=list(activity_totals.values()),
        savings_forecast=savings_forecast,
        gpa_forecast=gpa_forecast,
        projected_fitness=projected_fitness,
    )


@app.route("/user/goals/add", methods=["POST"])
@login_required
def add_goal():
    description = request.form.get("description", "").strip()
    if not description:
        flash("Please enter a goal description.")
        return redirect(url_for("user_dashboard"))

    target_date_value = request.form.get("target_date", "").strip()
    try:
        target_date = date.fromisoformat(target_date_value) if target_date_value else None
    except ValueError:
        flash("Please enter a valid target date.")
        return redirect(url_for("user_dashboard"))

    db.session.add(
        Goal(user_id=current_user.id, description=description, target_date=target_date)
    )
    db.session.commit()
    flash("Goal added.")
    return redirect(url_for("user_dashboard"))


def transaction_form_values():
    """Validate and return the fields shared by transaction create and edit forms."""
    transaction_date = date.fromisoformat(request.form.get("date", "").strip())
    category = request.form.get("category", "").strip()
    amount = float(request.form.get("amount", ""))
    transaction_type = request.form.get("type", "").strip()

    if not category or len(category) > 50:
        raise ValueError("Please enter a valid transaction category.")
    if not math.isfinite(amount) or amount <= 0:
        raise ValueError("Please enter an amount greater than zero.")
    if transaction_type not in {"income", "expense"}:
        raise ValueError("Please select a valid transaction type.")

    return {
        "date": transaction_date,
        "category": category,
        "amount": amount,
        "type": transaction_type,
    }


@app.route("/user/financial", methods=["GET", "POST"])
@login_required
def user_financial():
    if request.method == "POST":
        try:
            transaction = Transaction(
                user_id=current_user.id,
                **transaction_form_values(),
            )
        except (ValueError, TypeError):
            flash("Please enter a valid date, category, amount, and transaction type.")
        else:
            db.session.add(transaction)
            db.session.commit()
            flash("Transaction added.")
        return redirect(url_for("user_financial"))

    transactions = Transaction.query.filter_by(user_id=current_user.id).order_by(
        Transaction.date.desc(), Transaction.id.desc()
    ).all()
    income = sum(tx.amount for tx in transactions if tx.type == "income")
    expenses = sum(tx.amount for tx in transactions if tx.type == "expense")
    expense_categories = defaultdict(float)
    monthly_expenses = defaultdict(float)
    expense_count = 0
    for transaction in transactions:
        if transaction.type == "expense":
            expense_count += 1
            expense_categories[transaction.category] += transaction.amount
            monthly_expenses[transaction.date.strftime("%Y-%m")] += transaction.amount

    expense_months = sorted(monthly_expenses)
    forecast = (
        forecast_values(
            [monthly_expenses[month] for month in expense_months[-6:]],
            6,
        )
        if expense_count >= 3
        else []
    )
    return render_template(
        "user/financial.html",
        metrics={
            "income": income,
            "expenses": expenses,
            "net": income - expenses,
            "rate": (income - expenses) / income * 100 if income else 0,
        },
        transactions=transactions,
        categories=list(expense_categories),
        category_amounts=list(expense_categories.values()),
        forecast=forecast,
    )


@app.route("/user/financial/edit", methods=["POST"])
@login_required
def edit_transaction():
    try:
        transaction_id = int(request.form.get("id", ""))
    except ValueError:
        flash("Please select a valid transaction to edit.")
        return redirect(url_for("user_financial"))
    transaction = Transaction.query.filter_by(
        id=transaction_id, user_id=current_user.id
    ).first_or_404()
    try:
        values = transaction_form_values()
    except (ValueError, TypeError):
        flash("Please enter a valid date, category, amount, and transaction type.")
    else:
        for field, value in values.items():
            setattr(transaction, field, value)
        db.session.commit()
        flash("Transaction updated.")
    return redirect(url_for("user_financial"))


@app.route("/user/financial/<int:transaction_id>/delete", methods=["POST"])
@login_required
def delete_transaction(transaction_id):
    transaction = Transaction.query.filter_by(
        id=transaction_id, user_id=current_user.id
    ).first_or_404()
    db.session.delete(transaction)
    db.session.commit()
    flash("Transaction deleted.")
    return redirect(url_for("user_financial"))

@app.route("/user/study")
@login_required
def user_study():
    logs = StudyLog.query.filter_by(user_id=current_user.id).order_by(
        StudyLog.date.desc(), StudyLog.id.desc()
    ).all()
    total_hours = sum(log.hours for log in logs)
    average_productivity = (
        sum(log.productivity_score for log in logs) / len(logs) if logs else 0
    )
    gpa = (
        min(4.0, max(1.0, 2.0 + total_hours / 100 + average_productivity / 100))
        if logs
        else 3.0
    )

    today = date.today()
    chart_dates = [today - timedelta(days=offset) for offset in range(13, -1, -1)]
    daily_hours = defaultdict(float)
    daily_productivity = defaultdict(list)
    subject_hours = defaultdict(float)
    study_dates = set()
    for log in logs:
        subject_hours[log.subject] += log.hours
        study_dates.add(log.date)
        if log.date in chart_dates:
            daily_hours[log.date] += log.hours
            daily_productivity[log.date].append(log.productivity_score)

    study_streak = 0
    streak_date = today
    while streak_date in study_dates:
        study_streak += 1
        streak_date -= timedelta(days=1)

    return render_template(
        "user/study.html",
        logs=logs,
        total_hours=total_hours,
        average_productivity=average_productivity,
        study_streak=study_streak,
        gpa=gpa,
        chart_labels=[day.strftime("%d %b") for day in chart_dates],
        chart_hours=[daily_hours[day] for day in chart_dates],
        chart_productivity=[
            (
                sum(daily_productivity[day]) / len(daily_productivity[day])
                if daily_productivity[day]
                else None
            )
            for day in chart_dates
        ],
        subjects=list(subject_hours),
        subject_hours=list(subject_hours.values()),
        today=today.isoformat(),
        updated_at=datetime.now(),
    )


def study_form_values():
    """Validate the fields shared by study-session creation and editing."""
    session_date = date.fromisoformat(request.form.get("date", "").strip())
    hours = float(request.form.get("hours", ""))
    subject = request.form.get("subject", "").strip()
    productivity_score = int(request.form.get("productivity_score", ""))

    if not subject or len(subject) > 100:
        raise ValueError("Please enter a subject with no more than 100 characters.")
    if not math.isfinite(hours) or not 0 < hours <= 24:
        raise ValueError("Study hours must be greater than 0 and no more than 24.")
    if not 0 <= productivity_score <= 100:
        raise ValueError("Productivity must be between 0 and 100.")

    return {
        "date": session_date,
        "hours": hours,
        "subject": subject,
        "productivity_score": productivity_score,
    }


@app.route("/user/study", methods=["POST"])
@login_required
def add_study_log():
    try:
        log = StudyLog(user_id=current_user.id, **study_form_values())
    except (ValueError, TypeError):
        flash("Please enter a valid date, hours, subject, and productivity score.")
    else:
        db.session.add(log)
        db.session.commit()
        flash("Study session logged.")
    return redirect(url_for("user_study"))


@app.route("/user/study/edit", methods=["POST"])
@login_required
def edit_study_log():
    try:
        log_id = int(request.form.get("id", ""))
    except ValueError:
        flash("Please select a valid study session to edit.")
        return redirect(url_for("user_study"))

    log = StudyLog.query.filter_by(id=log_id, user_id=current_user.id).first_or_404()
    try:
        values = study_form_values()
    except (ValueError, TypeError):
        flash("Please enter a valid date, hours, subject, and productivity score.")
    else:
        for field, value in values.items():
            setattr(log, field, value)
        db.session.commit()
        flash("Study session updated.")
    return redirect(url_for("user_study"))


@app.route("/user/study/<int:study_log_id>/delete", methods=["POST"])
@login_required
def delete_study_log(study_log_id):
    log = StudyLog.query.filter_by(
        id=study_log_id, user_id=current_user.id
    ).first_or_404()
    db.session.delete(log)
    db.session.commit()
    flash("Study session deleted.")
    return redirect(url_for("user_study"))


@app.route("/user/fitness")
@login_required
def user_fitness():
    logs = FitnessLog.query.filter_by(user_id=current_user.id).order_by(
        FitnessLog.date.desc(), FitnessLog.id.desc()
    ).all()
    total_minutes = sum(log.duration_min for log in logs)
    total_calories = sum(log.calories_burned or 0 for log in logs)
    daily_average = total_minutes / len({log.date for log in logs}) if logs else 0

    today = date.today()
    week_start = today - timedelta(days=today.weekday())
    chart_dates = [today - timedelta(days=offset) for offset in range(13, -1, -1)]
    daily_minutes = defaultdict(int)
    activity_minutes = defaultdict(int)
    fitness_dates = set()
    weekly_minutes = 0
    for log in logs:
        fitness_dates.add(log.date)
        daily_minutes[log.date] += log.duration_min
        if week_start <= log.date <= today:
            weekly_minutes += log.duration_min
        if log.activity:
            activity_minutes[log.activity] += log.duration_min

    fitness_streak = 0
    streak_date = today
    while streak_date in fitness_dates:
        fitness_streak += 1
        streak_date -= timedelta(days=1)

    return render_template(
        "user/fitness.html",
        logs=logs,
        total_minutes=total_minutes,
        total_calories=total_calories,
        daily_average=daily_average,
        fitness_streak=fitness_streak,
        chart_labels=[day.strftime("%Y-%m-%d") for day in chart_dates],
        chart_minutes=[daily_minutes[day] for day in chart_dates],
        activities=list(activity_minutes),
        activity_minutes=list(activity_minutes.values()),
        weekly_minutes=weekly_minutes,
        today=today.isoformat(),
        updated_at=datetime.now(),
    )


def fitness_form_values():
    """Validate the fields shared by fitness-activity creation and editing."""
    activity_date = date.fromisoformat(request.form.get("date", "").strip())
    activity = request.form.get("activity", "").strip()
    duration_min = int(request.form.get("duration_min", ""))
    calories_burned = int(request.form.get("calories_burned", "0"))

    if not activity or len(activity) > 50:
        raise ValueError("Please enter an activity with no more than 50 characters.")
    if not 1 <= duration_min <= 1440:
        raise ValueError("Activity duration must be between 1 and 1440 minutes.")
    if not 0 <= calories_burned <= 10000:
        raise ValueError("Calories burned must be between 0 and 10000.")

    return {
        "date": activity_date,
        "activity": activity,
        "duration_min": duration_min,
        "calories_burned": calories_burned,
    }


@app.route("/user/fitness", methods=["POST"])
@login_required
def add_fitness_log():
    try:
        log = FitnessLog(user_id=current_user.id, **fitness_form_values())
    except (ValueError, TypeError):
        flash("Please enter a valid date, activity, duration, and calorie count.")
    else:
        db.session.add(log)
        db.session.commit()
        flash("Fitness activity logged.")
    return redirect(url_for("user_fitness"))


@app.route("/user/fitness/edit", methods=["POST"])
@login_required
def edit_fitness_log():
    try:
        log_id = int(request.form.get("id", ""))
    except ValueError:
        flash("Please select a valid fitness activity to edit.")
        return redirect(url_for("user_fitness"))

    log = FitnessLog.query.filter_by(id=log_id, user_id=current_user.id).first_or_404()
    try:
        values = fitness_form_values()
    except (ValueError, TypeError):
        flash("Please enter a valid date, activity, duration, and calorie count.")
    else:
        for field, value in values.items():
            setattr(log, field, value)
        db.session.commit()
        flash("Fitness activity updated.")
    return redirect(url_for("user_fitness"))


@app.route("/user/fitness/<int:fitness_log_id>/delete", methods=["POST"])
@login_required
def delete_fitness_log(fitness_log_id):
    log = FitnessLog.query.filter_by(
        id=fitness_log_id, user_id=current_user.id
    ).first_or_404()
    db.session.delete(log)
    db.session.commit()
    flash("Fitness activity deleted.")
    return redirect(url_for("user_fitness"))


@app.route("/user/simulation", methods=["GET", "POST"])
@login_required
def user_simulation():
    result = None
    current_values = {
        "savings": current_user.savings or 0,
        "gpa": user_gpa(current_user),
        "fitness": sum(log.duration_min for log in current_user.fitness_logs),
    }
    if request.method == "POST":
        scenario = request.form.get("scenario", "").strip()
        try:
            amount = float(request.form.get("amount", ""))
        except (TypeError, ValueError):
            amount = math.nan

        scenario_limits = {
            "save": (100, 100000),
            "invest": (100, 100000),
            "spend": (0, 50000),
            "study_more": (0, 40),
            "exercise": (0, 120),
        }
        limits = scenario_limits.get(scenario)
        if limits is None or not math.isfinite(amount) or not limits[0] <= amount <= limits[1]:
            flash("Please choose a valid scenario and enter an amount within its allowed range.")
            return redirect(url_for("user_simulation"))

        result = simulate_scenario(current_user, scenario, amount)
        result["scenario"] = scenario
        db.session.add(
            SimulationHistory(
                user_id=current_user.id,
                scenario=scenario,
                amount=amount,
                result_desc=result["desc"],
                result_impact=result["impact"],
                result_risk=result["risk"],
                result_rec=result["rec"],
                projected=result["projected"],
                current=result["current"],
            )
        )
        db.session.commit()

    history = SimulationHistory.query.filter_by(user_id=current_user.id).order_by(
        SimulationHistory.timestamp.desc(), SimulationHistory.id.desc()
    ).all()
    return render_template(
        "user/simulation.html",
        user=current_user,
        result=result,
        current_values=current_values,
        history=history,
    )


@app.route("/user/simulation/clear", methods=["POST"])
@login_required
def clear_simulation_history():
    SimulationHistory.query.filter_by(user_id=current_user.id).delete()
    db.session.commit()
    flash("Simulation history cleared.")
    return redirect(url_for("user_simulation"))


@app.route("/user/assistant", methods=["GET", "POST"])
@login_required
def user_assistant():
    if request.args.get("new"):
        session["assistant_new_chat"] = True
        return redirect(url_for("user_assistant"))

    all_messages = ChatMessage.query.filter_by(user_id=current_user.id).order_by(
        ChatMessage.timestamp.desc(), ChatMessage.id.desc()
    ).all()
    selected_message = None
    load_id = request.args.get("load")
    if load_id:
        try:
            message_id = int(load_id)
        except ValueError:
            message_id = None
        if message_id is None:
            flash("That conversation could not be found.")
        else:
            selected_message = ChatMessage.query.filter_by(
                id=message_id, user_id=current_user.id
            ).first()
            if selected_message is None:
                flash("That conversation could not be found.")

    if request.method == "POST":
        query = request.form.get("query", "").strip()
        if not query:
            flash("Please enter a question.")
            return redirect(url_for("user_assistant"))
        if len(query) > 2000:
            flash("Please keep your question under 2,000 characters.")
            return redirect(url_for("user_assistant"))

        response = None
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                import google.generativeai as genai

                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-2.5-flash")
                prompt = (
                    "You are a helpful personal decision assistant. Give concise, "
                    "practical guidance using the user's profile where relevant. "
                    "Do not claim certainty for financial or health outcomes.\n\n"
                    f"User: {current_user.username}\n"
                    f"Monthly income: ₹{current_user.monthly_income or 0:,.0f}\n"
                    f"Monthly expenses: ₹{current_user.monthly_expenses or 0:,.0f}\n"
                    f"Savings: ₹{current_user.savings or 0:,.0f}\n"
                    f"Estimated GPA: {user_gpa(current_user):.2f}/4.0\n"
                    f"Study hours logged: {sum(log.hours for log in current_user.study_logs):g}\n"
                    f"Fitness minutes logged: {sum(log.duration_min for log in current_user.fitness_logs)}\n"
                    f"Question: {query}"
                )
                generated = model.generate_content(prompt)
                response = (generated.text or "").strip()
                if not response:
                    raise ValueError("The AI provider returned an empty response.")
            except Exception:
                app.logger.exception("AI assistant provider request failed")
                flash("The AI service is unavailable; personalized local guidance was shown instead.")
                response = local_assistant_response(query, current_user)
        else:
            flash("AI is not configured; personalized local guidance is being used.")
            response = local_assistant_response(query, current_user)

        message = ChatMessage(
            user_id=current_user.id,
            question=query,
            response=response,
        )
        db.session.add(message)
        db.session.commit()
        session.pop("assistant_new_chat", None)
        return redirect(url_for("user_assistant", load=message.id))

    if selected_message:
        messages = [selected_message]
    elif session.pop("assistant_new_chat", False):
        messages = []
    else:
        messages = list(reversed(all_messages))
    return render_template(
        "user/assistant.html",
        user=current_user,
        all_messages=all_messages,
        messages=messages,
    )


@app.route("/settings")
@login_required
def settings():
    return simple_page("Settings", "<p>Settings page.</p>")


@app.route("/admin/dashboard")
@login_required
def admin_dashboard():
    return simple_page("Admin dashboard", "<p>Admin dashboard.</p>")


@app.route("/admin/users")
@login_required
def admin_users():
    return simple_page("Admin users", "<p>Admin users.</p>")


@app.route("/privacy")
def privacy():
    return render_template("privacy.html")


@app.route("/terms")
def terms():
    return render_template("terms.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/about")
def about():
    return render_template("about.html")


with app.app_context():
    db.create_all()


if __name__ == "__main__":
    app.run(debug=True)
