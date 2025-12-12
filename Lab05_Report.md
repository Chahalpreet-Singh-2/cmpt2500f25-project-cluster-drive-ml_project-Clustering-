Group Name: Cluster Drive

Group Members:  1. Chahalpreet Singh : 3096522
                2. Parminder Singh : 3095849
                3. Rajveer Singh : 3075355
                4. Arpandeep Kaur : 3097220

  # 1. Metrics Implemented and Why :- 
  a) ml_predictions_total (Counter):
  * Tracks total predictions, separated by labels (model_version, status, prediction_result).
  *  Useful to monitor traffic, successes, and errors.
  b) ml_prediction_duration_seconds (Histogram):
  * Measures latency for each prediction request.
  * Helps identify performance bottlenecks.
  c) app_memory_usage_bytes (Gauge):
  * Shows how much memory the service consumes over time.
  d) app_cpu_usage_percent (Gauge):
  * Helps detect high CPU usage or infinite loops.
  e) model_loaded (Gauge):
  * Confirms whether the model is loaded properly on startup.

  These metrics were chosen because they provide visibility into: * API reliability
                                                                  * Error rates
                                                                  * Performance issues
                                                                  * Resource consumption

 # 2. How Monitoring Helps in ML Systems :-
 a) Monitoring is essential in ML systems because models can drift, degrade, or fail silently.
 b) Prometheus + Grafana helps with: * Detecting spikes in error rate
                                     * Identifying slow predictions
                                     * Catching failed model loads
                                     * Alerting when system performance drops
                                     * Observing traffic patterns
                                     * Understanding resource usage (CPU / RAM)
  With these visual dashboards in Grafana, engineers can quickly understand system status and respond before end users are impacted.
   
 # 3 Challenges Encountered and their solutions:- 
 a) Challenge 1 :- Flask endpoint duplication error
    Solution:- Removed the duplicate /health endpoint. After restarting Docker Compose, the API started correctly.

  b) Challenge 2 :- The service kept restarting because Prometheus metrics were not initialized correctly.
     Solution:- Copied the exact Prometheus instrumentation setup from lab instructions and ensured all counters, histograms, and gauges had unique names.

  c) Challenge 3 :- At first, no data appeared in Prometheus UI.
     Solution:- Verified port 5000 was forwarded-mapping, scrape_configs in prometheus.yml used ml-app:5000 and generated traffic using curl loops.

  d) Challenege 4:- The DemoHighErrorCount alert didn’t trigger at first.
     Solution:- Ensured to sent multiple invalid prediction requests and waited the required 10–15 seconds.

  # Conclusion:- 
  This lab demonstrated the importance of monitoring in ML systems. By adding Prometheus metrics, configuring alerts, and visualizing results in Grafana, we created a fully observable ML API that can detect errors, performance issues, and failures proactively.
 
