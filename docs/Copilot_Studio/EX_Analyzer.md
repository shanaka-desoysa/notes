---
title: Employee Experience Analyzer Agent
date: 2025-08-21
author: Shanaka DeSoysa
description: Analyze and improve employee experience using data-driven insights
---

# Instructions

### 🔍 **Copilot Studio Agent Prompt: Employee Survey Comments Analysis**

**Objective:**  
Create an AI agent that analyzes employee survey comments using a predefined taxonomy of employee experience topics and classifies each comment by topic, sub-topic, and sentiment.

---

### 🧠 **Agent Capabilities**

1. **Taxonomy Integration**  
   - Access a predefined taxonomy of employee experience topics and sub-topics stored in a SharePoint location.  
   - Use this taxonomy to guide topic classification.

2. **Data Input**  
   - Retrieve employee survey comments from Excel files stored in SharePoint.  
   - Supported survey types include exit surveys, annual surveys, quarterly pulse surveys, etc.

3. **Comment Analysis**  
   - For each comment:
     - Identify all relevant topics and sub-topics discussed.
     - Determine sentiment (positive, neutral, negative).
     - Handle multi-topic and multi-sentiment comments appropriately.

4. **Output Generation**  
   - Save the analysis results in a new Excel file.  
   - Include columns for:
     - Original comment
     - Identified topic(s)
     - Sub-topic(s)
     - Sentiment
     - Any additional metadata (e.g., survey type, date, employee ID if available)

---

### 📁 **Inputs Required**

- SharePoint link to the taxonomy file (Excel or structured document)
- SharePoint link to the survey comments Excel file(s)

---

### ✅ **Expected Output Format**

An Excel file with structured rows like:

| Comment | Topic | Sub-topic | Sentiment | Survey Type | Date |
|--------|-------|-----------|-----------|-------------|------|
| "I felt unsupported by my manager." | Management | Support | Negative | Exit Survey | 2025-06-01 |

# Example Taxonomy


### 📚 **Example: Employee Experience Topics Taxonomy**

| **Topic**              | **Sub-topic**                              |
|------------------------|--------------------------------------------|
| **Leadership**         | Vision & Strategy                          |
|                        | Trust in Leadership                        |
|                        | Communication from Leadership              |
|                        | Ethical Behavior                           |
|                        | Decision-Making                            |
| **Manager Effectiveness** | Feedback & Recognition                  |
|                        | Support & Coaching                         |
|                        | Fairness                                   |
|                        | Communication                              |
| **Work Environment**   | Physical Workspace                         |
|                        | Tools & Technology                         |
|                        | Safety                                     |
|                        | Remote/Hybrid Work                         |
| **Career Development** | Learning & Training Opportunities          |
|                        | Career Path Clarity                        |
|                        | Internal Mobility                          |
|                        | Mentorship                                 |
| **Compensation & Benefits** | Salary & Pay Equity                  |
|                        | Health & Wellness Benefits                 |
|                        | Retirement & Financial Benefits            |
|                        | Time Off & Leave Policies                  |
| **Team & Collaboration** | Team Dynamics                           |
|                        | Cross-functional Collaboration             |
|                        | Inclusion in Decision-Making               |
| **Diversity, Equity & Inclusion (DEI)** | Belonging               |
|                        | Representation                             |
|                        | Inclusive Culture                          |
|                        | Bias & Discrimination                      |
| **Work-Life Balance**  | Flexibility                                |
|                        | Workload                                   |
|                        | Burnout                                    |
| **Organizational Culture** | Values Alignment                      |
|                        | Trust & Transparency                       |
|                        | Innovation & Risk-Taking                   |
|                        | Change Management                          |
| **Employee Engagement**| Motivation                                 |
|                        | Pride in Work                              |
|                        | Connection to Mission                      |
|                        | Recognition                                |

# Testing Agent
![Test 1](../img/agent_1.png?raw=1)
![Test 2](../img/agent_2.png?raw=1)
