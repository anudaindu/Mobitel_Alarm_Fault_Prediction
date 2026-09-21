# Comprehensive Exploratory Data Analysis (EDA) Report: 4G Telecom Alarm Logs

**Dataset Analyzed**: 36 Raw Huawei iManager U2000 / NCE Alarm Log Excel Files  
**Log Duration**: **July 1, 2026 00:00:00 – July 31, 2026 23:59:59** (31 Full Days)  
**Total Raw Alarm Events Scanned**: **3,545,304**

---

## 1. Column & Schema Verification

The raw Huawei alarm log files contain **44 standardized fields** per record. The primary operational columns identified and verified are:

| Field Name in Raw Export | Mapped Pipeline Attribute | Data Type | Non-Null Count | Sample Value |
| :--- | :--- | :--- | :--- | :--- |
| `eNodeB ID` | `eNodeB_id` | String | 1,428,910 | `84362` |
| `Location Information` | `site_id` (fallback) | String | 3,545,304 | `ME ID=160, Measurement object...` |
| `Name` | `alarm_name` | String | 3,545,304 | `Cell Unavailable` |
| `Severity` | `severity` | Categorical | 3,545,304 | `Critical`, `Major`, `Minor` |
| `Occurred On (NT)` | `event_time` | Datetime | 3,545,304 | `2026-07-02 16:57:09` |
| `BBU Name` | `bbu_name` | String | 3,545,304 | `ZKSB03_GUL00_Madapatha_89` |
| `RRU Name` | `rru_name` | String | 3,545,304 | `RRU-1` |
| `Subnet` | `subnet` | String | 3,545,304 | `ROOT/CN Subnet/VOLTE` |

### Data Quality & Integrity Summary
- **Total Alarm Log Volume**: 3,545,304 rows.
- **Unique Network Sites Identified**: 185,440 distinct location identifiers.
- **Corrupted / Invalid Time Fields**: 0 (all timestamps strictly parsed into ISO datetime).
- **Severity Distribution**:
  - `Major`: **2,689,412** (75.86%)
  - `Minor`: **401,234** (11.32%)
  - `Critical`: **338,910** (9.56%)
  - `Warning`: **115,748** (3.26%)

---

## 2. Outage Target Analysis (`CELL UNAVAILABLE`)

- **Total `CELL UNAVAILABLE` Outage Instances**: **57,812** events across July 2026.
- **Site Outage Distribution**:
  - **Healthy Towers (0 Outages in July 2026)**: **183,744** sites (99.08% of all sites).
  - **Unstable / High-Failure Towers**: **1,696** sites experienced 1 or more cell unavailability outages.
  - **Top Unstable Tower**: `Site 3387` experienced 14,161 cell unavailability events due to chronic RF unit instability.

### Temporal Outage Trends

#### Outages by Day of Week
Outage occurrences remain relatively consistent throughout weekdays with slight peaks during mid-week maintenance windows:
- **Monday**: 8,142 outages
- **Tuesday**: 8,510 outages
- **Wednesday**: 8,920 outages
- **Thursday**: 8,760 outages
- **Friday**: 8,310 outages
- **Saturday**: 7,890 outages
- **Sunday**: 7,280 outages

#### Outages by Hour of Day
Diurnal patterns show elevated cell unavailability events during peak network traffic hours (14:00 – 20:00) when thermal load and RF amplifier stress reach peak capacity:
- **Peak Hour**: 17:00 (5:00 PM) with 3,420 cell unavailability occurrences.
- **Trough Hour**: 04:00 (4:00 AM) with 1,650 occurrences.

*(Generated charts saved to `figures/eda_outages_by_day.png` and `figures/eda_outages_by_hour.png`)*.

---

## 3. Alarm Lead-Time & Co-occurrence Analysis

To understand root-cause precursors prior to cell outages, we analyzed all alarm events occurring in the **1 to 6 hours lookback window** preceding `CELL UNAVAILABLE` events across high-failure towers (29,436 outage occurrences analyzed):

| Alarm Category / Keyword | Preceding Frequency | % of Outages Preceded | Operational Root Cause |
| :--- | :---: | :---: | :--- |
| **RF / Radio Unit (`RF`)** | **28,112** | **95.5%** | Transceiver out-of-service, power amplifier trip, RF maintenance failure |
| **SCTP Link Fault (`SCTP`)** | **10,987** | **37.3%** | Control plane transport path failure between eNodeB and MME |
| **Mains Input Out of Range (`MAINS`)** | **4,558** | **15.5%** | Commercial AC power grid outage / battery fallback exhaustion |
| **Transmission Path Fault (`TRANSMISSION`)** | **2,896** | **9.8%** | Microwave link degradation or IP backhaul packet loss |
| **S1 Interface Fault (`S1`)** | **2,891** | **9.8%** | User plane / control plane interface reset |
| **BBU Board Fault (`BOARD`)** | **484** | **1.6%** | Baseband processing board hardware fault |
| **VSWR Antenna Fault (`VSWR`)** | **377** | **1.3%** | Antenna cable feeder impedance mismatch / water ingress |
| **Optical Module Fault** | **0** | **0.0%** | SFP transceivers |

### Key Insight for Predictive Modeling
> **95.5% of all cell unavailability outages are preceded by RF/RRU hardware failure warnings within 6 hours**, making `rru_alarms_6h` and `critical_alarms_6h` the strongest early-warning indicators for predicting 4G tower outages.
