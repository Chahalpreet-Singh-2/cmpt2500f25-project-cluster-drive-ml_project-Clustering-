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

 # 2.  How Monitoring Helps in ML Systems :-
 a) Monitoring is essential in ML systems because models can drift, degrade, or fail silently.
 b) Prometheus + Grafana helps with: * Detecting spikes in error rate
                                     * Identifying slow predictions
                                     * Catching failed model loads
                                     * Alerting when system performance drops
                                     * Observing traffic patterns
                                     * Understanding resource usage (CPU / RAM)
  With these visual dashboards in Grafana, engineers can quickly understand system status and respond before end users are impacted.
   

