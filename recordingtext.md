Intro (30 seconds)
Hi, my name is Armen, and this project is an AI-powered dairy cow monitoring system that I built as an end-to-end prototype, inspired by real farm monitoring platforms used in the veterinary and animal-health industry.

The idea came from conversations with my family, who work in veterinary and livestock production. Modern farms already collect a lot of data from cows — things like milk yield, feed intake, temperature, activity, and environmental conditions — but turning that data into early, actionable insight is still a challenge.

What the system does (45 seconds)
In this system, each cow has a unique ID, similar to an RFID chip or barcode used on farms. When a cow is scanned, the system pulls together its recent data and compares actual performance against an expected milk baseline based on breed and stage of lactation.

But instead of looking at milk alone, the system combines multiple signals:
feed intake, body temperature, rumination, activity, and environmental heat stress.

Based on these signals, the system classifies each cow as OK, WARNING, or FLAG, and it explains why — for example, reduced feed intake, elevated temperature, or declining rumination.

Why this matters (40 seconds)
In real farm settings, production drops often happen after a health or nutrition problem has already started. By the time milk drops, the issue may already be serious.

So the goal of this project is not just to detect problems, but to detect them early — before production collapses.

Early-warning AI model (1–1.5 minutes)
To do that, I added a machine-learning model that predicts whether a cow is likely to become a high-risk case within the next three days.

This is framed as a classification problem. The model looks at short-term trends, like declining milk, feed intake drops, increasing temperature, reduced rumination, and heat stress indicators.

The output is a probability score — for example, an 82% chance that this cow will become a flagged case in the near future.

This allows the system to surface cows that look normal today, but are trending in a dangerous direction.

In a real deployment, this kind of early warning could allow farmers or veterinarians to intervene earlier — adjusting nutrition, managing heat stress, or performing health checks before losses occur.

System design & tech stack (45 seconds)
From a technical perspective, this is a full-stack system.

The backend is built with FastAPI and SQLAlchemy, exposing a clean API for scanning cows, retrieving herd-level alerts, logging audits, and running machine-learning predictions.

The frontend is a Streamlit dashboard that provides a herd overview, individual cow profiles, time-series charts, and early-warning risk scores.

The entire system is containerized using Docker Compose, so it can be run with a single command, which makes it easy to demo and deploy.

Why this project matters (30 seconds)
This project demonstrates how AI can be applied responsibly in agriculture and animal health — combining domain knowledge, explainable rules, and machine learning to support better decision-making.

It’s designed as a realistic prototype that could be extended with real sensor data, real farm integrations, and more advanced models in the future.

Thanks for watchin