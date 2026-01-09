## 📊 Dataset Description

| Column Name                     | Description                                                                 |
|--------------------------------|-----------------------------------------------------------------------------|
| `Date`                         | Date of the booking                                                         |
| `Time`                         | Time of the booking                                                         |
| `Booking ID`                   | Unique identifier for each ride booking                                    |
| `Booking Status`               | Status of booking (Completed, Cancelled by Customer, Cancelled by Driver) |
| `Customer ID`                  | Unique identifier for customers                                             |
| `Vehicle Type`                 | Type of vehicle (Go Mini, Go Sedan, Auto, eBike/Bike, UberXL, Premier Sedan) |
| `Pickup Location`              | Starting location of the ride                                               |
| `Drop Location`                | Destination location of the ride                                            |
| `Avg VTAT`                     | Average Vehicle Time at Arrival                                             |
| `Avg CTAT`                     | Average Customer Time at Arrival                                            |
| `Cancelled Rides by Customer`  | Customer-initiated cancellation flag                                        |
| `Reason for cancelling by Customer` | Reason for customer cancellation                                      |
| `Cancelled Rides by Driver`    | Driver-initiated cancellation flag                                          |
| `Driver Cancellation Reason`   | Reason for driver cancellation                                              |
| `Incomplete Rides`             | Incomplete ride flag                                                        |
| `Incomplete Rides Reason`      | Reason for incomplete rides                                                 |
| `Booking Value`                | Total fare amount for the ride                                              |
| `Ride Distance`                | Distance covered during the ride (in km)                                    |
| `Driver Ratings`               | Rating given to driver (1–5 scale)                                          |
| `Customer Rating`              | Rating given by customer (1–5 scale)                                        |
| `Payment Method`               | Method used for payment (UPI, Cash, Credit Card, Uber Wallet, Debit Card)  |




Group,Features to Keep
External/Weather,"temp, precip, windspeed, visibility"
Calendar/Time,"day_of_month, pickup_hour, day_of_week, is_weekend, is_holiday"
Special Events,"is_special_event, is_payday"
Traffic,"city_congestion_index, zone_congestion_index"
History (Lags),"pickups_lag_1h, pickups_lag_24h, city_pickups_lag_1h, city_pickups_lag_24h, neighbor_pickups_lag_1h"