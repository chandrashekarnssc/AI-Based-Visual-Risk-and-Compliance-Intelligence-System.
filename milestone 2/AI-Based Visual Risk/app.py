"""Visual Risk Flask application."""

from collections import defaultdict
from datetime import date, datetime, timedelta
import html
import math
import os

from dotenv import load_dotenv
from flask import Flask, flash, redirect, render_template, render_template_string, request, url_for
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
    for offset in range(7):
        if today - timedelta(days=offset) in study_dates:
            study_streak += 1
        else:
            break
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
    return simple_page("Study tracker", "<p>Your study tracker is ready for the restored page templates.</p>")


@app.route("/user/fitness")
@login_required
def user_fitness():
    return simple_page("Fitness tracker", "<p>Your fitness tracker is ready for the restored page templates.</p>")


@app.route("/user/simulation")
@login_required
def user_simulation():
    return simple_page("Simulation", "<p>Simulation is available after its page template is restored.</p>")


@app.route("/user/assistant")
@login_required
def user_assistant():
    return simple_page("AI assistant", "<p>The AI assistant is available after its page template is restored.</p>")


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
