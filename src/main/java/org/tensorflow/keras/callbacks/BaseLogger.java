/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=======================================================================*/
package org.tensorflow.keras.callbacks;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Callback that accumulates epoch averages of metrics.
 * 
 * @author jbclarke
 */
public class BaseLogger extends Callback {
    
    /**
     * List of string names of metrics that should not be averaged over an epoch. 
     * Metrics in this list will be logged as-is in on_epoch_end. 
     * All others will be averaged in on_epoch_end.
     */
    public final List<String> statefulMetrics;
    
    private int seen;
    private final Map<String, Number> totals = new HashMap<>();
    
    /**
     * Create a new BaseLogger
     */
    public BaseLogger() {
        this(Collections.EMPTY_LIST);
    }
    
    /**
     * Create a BaseLogger
     * 
     * @param params Training parameters
     */
    public BaseLogger(Map<String, Object> params) {
        this(params, null, Collections.EMPTY_LIST);
    }
    
    /**
     * Create a BaseLogger
     * 
     * @param params Training parameters
     * @param model Reference of the model being trained.
     */
    public BaseLogger(Map<String, Object> params, Object model) {
        this(params, model, Collections.EMPTY_LIST);
    }
    
    /**
     * Create a new BaseLogger
     * 
     * @param statefulMetrics string names of metrics that
     * should *not* be averaged over an epoch.
     * Metrics in this list will be logged as-is in `on_epoch_end`.
     *  All others will be averaged in `on_epoch_end`.
     */
    public BaseLogger(String... statefulMetrics) {
        this(Arrays.asList(statefulMetrics));
    }
    
    /**
     * Create a new BaseLogger
     * 
     * @param statefulMetrics  names of metrics that
     * should *not* be averaged over an epoch.
     * Metrics in this list will be logged as-is in `on_epoch_end`.
     *  All others will be averaged in `on_epoch_end`.
     */
    public BaseLogger(List<String> statefulMetrics) {
        super();
        this.statefulMetrics = statefulMetrics;
    }
     /**
     * Create a BaseLogger
     * 
     * @param params Training parameters
     */
    public BaseLogger( Map<String, Object> params, String... statefulMetrics) {
        this( params, null, Arrays.asList(statefulMetrics));
    }
    
    /**
     * Create a BaseLogger
     * 
     * @param params Training parameters
     * @param model Reference of the model being trained.
     * @param statefulMetrics names of metrics that
     * should *not* be averaged over an epoch.
     * Metrics in this list will be logged as-is in `on_epoch_end`.
     *  All others will be averaged in `on_epoch_end`.
     */
    public BaseLogger(Map<String, Object> params, Object model, String... statefulMetrics) {
        this( params, model, Arrays.asList(statefulMetrics));
    }
    /**
     * Create a BaseLogger
     * 
     * @param params Training parameters
     * @param statefulMetrics names of metrics that
     * should *not* be averaged over an epoch.
     * Metrics in this list will be logged as-is in `on_epoch_end`.
     *  All others will be averaged in `on_epoch_end`.
     */
    public BaseLogger(Map<String, Object> params, List<String> statefulMetrics ) {
        this(params, null, statefulMetrics);
    }
    
    /**
     * Create a BaseLogger
     * 
     * @param params Training parameters
     * @param model Reference of the model being trained.
     * @param statefulMetrics names of metrics that
     * should *not* be averaged over an epoch.
     * Metrics in this list will be logged as-is in `on_epoch_end`.
     * All others will be averaged in `on_epoch_end`.
     */
    public BaseLogger(Map<String, Object> params, Object model, List<String> statefulMetrics ) {
        super(params, model);
        this.statefulMetrics = statefulMetrics;
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public void onEpochBegin(int epoch, Map<String, Number> logs) {
        this.seen = 0;
        totals.clear();
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public void onEpochEnd(int epoch, Map<String, Number> logs) {
        if(logs != null) {
            if(this.params.containsKey("metrics")) {
                List<String> metrics = (List<String>) this.params.get("metrics");
                if(metrics != null) {
                    metrics.stream().filter(metric -> (this.totals.containsKey(metric))).forEachOrdered(metric -> {
                        if(this.statefulMetrics.contains(metric)) {
                            logs.put(metric, this.totals.get(metric));
                        }else {
                            
                            if(this.totals.get(metric) instanceof Number)
                                logs.put(metric, this.totals.get(metric).doubleValue() / (double)seen);
                        }
                    });
                }
            }
        }
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public void onTrainBatchEnd(int batch, Map<String, Number> logs) {
        logs = logs == null ? Collections.EMPTY_MAP : logs;
        Number batch_size = logs.getOrDefault("size", 0);
        Number num_steps = logs.getOrDefault("num_steps", 1);
        
        this.seen += batch_size.intValue() * num_steps.intValue();
        
        for(String key : logs.keySet()) {
            Number value = logs.get(key);
            if(this.statefulMetrics.contains(key)) {
                this.totals.put(key, value);
            } else {
                Number tVal = this.totals.get(key);
                double nVal = value.doubleValue() * batch_size.doubleValue();
                if(tVal != null) {
                    this.totals.put(key, 
                            tVal.doubleValue() + nVal);
                }else {
                    this.totals.put(key, nVal);
                }
            }
        }
    }
    
    
}
