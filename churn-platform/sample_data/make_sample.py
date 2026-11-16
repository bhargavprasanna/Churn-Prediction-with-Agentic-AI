"""Generates 4 CSVs with DIFFERENT schemas to exercise the domain-adaptive Data Agent."""
import numpy as np, pandas as pd
r = np.random.default_rng(7); N = 1000
def churn(z): return (1 / (1 + np.exp(-(z + r.normal(0, .8, N)))) > r.random(N)).astype(int)
def contact(i): return dict(email=[f"user{k}@example.com" if k % 9 else "" for k in range(N)], phone=[f"+9198765{k:05d}" for k in range(N)])
# 1) Generic SaaS/telecom
pay, sup, act, ten = r.poisson(.6, N), r.poisson(1.2, N), r.poisson(20, N), r.integers(1, 72, N); mc = np.round(r.normal(60, 25, N).clip(10), 2)
d = pd.DataFrame(dict(customer_id=[f"C{k:05d}" for k in range(N)], plan=r.choice(["Basic", "Standard", "Pro", "Enterprise"], N, p=[.35, .3, .25, .1]), tenure_months=ten, monthly_charges=mc,
    logins_30d=act, payment_issues=pay, support_contacts=sup, **contact(0), marketing_consent=r.choice(["yes", "no"], N, p=[.8, .2]), churn=churn(-.4 + .5*pay + .35*sup - .07*act - .02*ten)))
d.loc[r.choice(N, 12), "monthly_charges"] = np.nan; d.to_csv("customers.csv", index=False)
# 2) Spotify-like music streaming
lt, sp, ads, ao = r.gamma(3, 10, N), r.poisson(300, N), r.poisson(40, N), r.poisson(60, N)
pd.DataFrame(dict(UserID=[f"U{k}" for k in range(N)], Name=[f"Person {k}" for k in range(N)], Email=contact(0)["email"], SubscriptionType=r.choice(["Free", "Premium", "Family", "Student"], N, p=[.45, .3, .1, .15]),
    ListeningTimeHrs=lt.round(1), SongsPlayed=sp, AdsListened=ads, AppOpens=ao, IsChurned=churn(-.2 - .03*lt - .002*sp + .02*ads - .01*ao + 1.5))).to_csv("spotify_users.csv", index=False)
# 3) Netflix-like video streaming
mv, lg, dv, wt = r.poisson(25, N), r.poisson(30, N), r.integers(1, 6, N), r.gamma(3, 15, N)
pd.DataFrame(dict(customerID=[f"N{k}" for k in range(N)], plan=r.choice(["Basic", "Standard", "Premium"], N), movies_watched=mv, num_logins=lg, devices_used=dv, watch_time_hrs=wt.round(1),
    monthly_fee=r.choice([7.99, 13.99, 19.99], N), Churn=np.where(churn(1.2 - .04*mv - .03*lg - .02*wt - .1*dv) == 1, "Yes", "No"))).to_csv("netflix_users.csv", index=False)
# 4) Amazon-like e-commerce
od, ab, rt, ds, tk = r.poisson(12, N), r.uniform(0, 1, N), r.poisson(1.5, N), r.integers(1, 200, N), r.poisson(1, N)
pd.DataFrame(dict(cust_id=[f"A{k}" for k in range(N)], membership=r.choice(["Standard", "Prime"], N, p=[.6, .4]), orders_placed=od, cart_abandon_rate=ab.round(2), returns=rt, days_since_last_order=ds,
    avg_order_value=np.round(r.normal(45, 15, N).clip(5), 2), support_tickets=tk, Left=churn(-1.5 - .12*od + 1.2*ab + .25*rt + .012*ds + .3*tk) == 1)).to_csv("amazon_customers.csv", index=False)
d.drop(columns="churn").to_csv("customers_no_labels.csv", index=False); print("generated 5 CSVs")
