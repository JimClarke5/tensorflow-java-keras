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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.tensorflow.keras.utils.ProgressBar;

/**
 * Callback that prints metrics to stdout.
 */
public class ProgbarLogger extends Callback {

    public enum CountMode {
        steps, samples
    };

    private List<String> statefulMetrics;
    private CountMode mode;

    private int seen = 0;
    ProgressBar progbar = null;
    private Integer target = null;
    private boolean verbose = true;
    private int epochs = 1;
    private boolean calledInFit = false;

    /**
     * Create a ProgbarLogger
     */
    public ProgbarLogger() {
        this(null, null, CountMode.samples, (List<String>) null);
    }

    /**
     * Create a ProgbarLogger
     *
     * @param mode Whether the progress bar should count samples seen or steps
     * (batches) seen.
     */
    public ProgbarLogger(CountMode mode) {
        this(null, null, mode, (List<String>) null);
    }

    /**
     * Create a ProgbarLogger
     *
     * @param statefulMetrics names of metrics that should not be averaged over
     * an epoch. Metrics in this list will be logged as-is. All others will be
     * averaged over time (e.g. loss, etc). If not provided, defaults to the
     * Model's metrics.
     */
    public ProgbarLogger(List<String> statefulMetrics) {
        this(null, null, CountMode.samples, statefulMetrics);
    }

    /**
     * Create a ProgbarLogger
     *
     * @param statefulMetrics names of metrics that should not be averaged over
     * an epoch. Metrics in this list will be logged as-is. All others will be
     * averaged over time (e.g. loss, etc). If not provided, defaults to the
     * Model's metrics.
     */
    public ProgbarLogger(String... statefulMetrics) {
        this(null, null, CountMode.samples, Arrays.asList(statefulMetrics));
    }

    /**
     * Create a ProgbarLogger
     *
     * @param mode Whether the progress bar should count samples seen or steps
     * (batches) seen.
     * @param statefulMetrics names of metrics that should not be averaged over
     * an epoch. Metrics in this list will be logged as-is. All others will be
     * averaged over time (e.g. loss, etc). If not provided, defaults to the
     * Model's metrics.
     */
    public ProgbarLogger(CountMode mode, List<String> statefulMetrics) {
        this(null, null, mode, statefulMetrics);
    }

    /**
     * Create a ProgbarLogger
     *
     * @param mode Whether the progress bar should count samples seen or steps
     * (batches) seen.
     * @param statefulMetrics names of metrics that should not be averaged over
     * an epoch. Metrics in this list will be logged as-is. All others will be
     * averaged over time (e.g. loss, etc). If not provided, defaults to the
     * Model's metrics.
     */
    public ProgbarLogger(CountMode mode, String... statefulMetrics) {
        this(null, null, mode, Arrays.asList(statefulMetrics));
    }

    /**
     * Create a ProgbarLogger
     *
     * @param params Training parameters
     */
    public ProgbarLogger(Map<String, Object> params) {
        this(params, null, CountMode.samples, (List<String>) null);
    }

    /**
     * Create a ProgbarLogger
     *
     * @param params Training parameters
     * @param model Reference of the model being trained.
     */
    public ProgbarLogger(Map<String, Object> params, Object model) {
        this(params, model, CountMode.samples, (List<String>) null);
    }

    /**
     * Create a ProgbarLogger
     *
     * @param params Training parameters
     * @param model Reference of the model being trained.
     * @param mode Whether the progress bar should count samples seen or steps
     * (batches) seen.
     * @param statefulMetrics names of metrics that should not be averaged over
     * an epoch. Metrics in this list will be logged as-is. All others will be
     * averaged over time (e.g. loss, etc). If not provided, defaults to the
     * Model's metrics.
     */
    public ProgbarLogger(Map<String, Object> params, Object model, CountMode mode, String... statefulMetrics) {
        this(params, model, CountMode.samples, Arrays.asList(statefulMetrics));
    }

    /**
     * Create a ProgbarLogger
     *
     * @param params Training parameters
     * @param model Reference of the model being trained.
     * @param mode Whether the progress bar should count samples seen or steps
     * (batches) seen.
     * @param statefulMetrics names of metrics that should not be averaged over
     * an epoch. Metrics in this list will be logged as-is. All others will be
     * averaged over time (e.g. loss, etc). If not provided, defaults to the
     * Model's metrics.
     */
    public ProgbarLogger(Map<String, Object> params, Object model, CountMode mode, List<String> statefulMetrics) {
        super(params, model);
        this.mode = mode;
        this.statefulMetrics = statefulMetrics;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void setParams(Map<String, Object> params) {
        super.setParams(params);
        this.verbose = (Boolean) params.getOrDefault("verbose", false);
        this.epochs = (Integer) params.getOrDefault("epochs", 1);
        this.target = mode == CountMode.steps ? (Integer) params.get("steps")
                : (Integer) params.get("samples");
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void onTrainBegin(Map<String, Number> logs) {
        this.calledInFit = true;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void onTestBegin(Map<String, Number> logs) {
        if (!calledInFit) {
            this.resetProgbar();
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void onPredictBegin(Map<String, Number> logs) {
        this.resetProgbar();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void onEpochBegin(int epoch, Map<String, Number> logs) {
        this.resetProgbar();
        if (verbose && this.epochs > 1) {
            Logger.getLogger(ProgbarLogger.class.getName()).log(Level.INFO,
                    String.format("Epoch %d/%d", epoch + 1, this.epochs));
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void onTrainBatchEnd(int batch, Map<String, Number> logs) {
        this.updateBatchProgbar(logs);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void onTestBatchEnd(int batch, Map<String, Number> logs) {
        this.updateBatchProgbar(logs);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void onPredictBatchEnd(int batch, Map<String, Number> logs) {
        this.updateBatchProgbar(null); // Don't pass prediction results.
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void onEpochEnd(int epoch, Map<String, Number> logs) {
        this.finalizeProgbar(logs);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void onTestEnd(Map<String, Number> logs) {
        if (!this.calledInFit) {
            this.finalizeProgbar(logs);
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void onPredictEnd(Map<String, Number> logs) {
        this.finalizeProgbar(logs);
    }

    private void resetProgbar() {
        this.seen = 0;
        this.progbar = null;
    }

    private void updateBatchProgbar(Map<String, Number> logs) {
        if (this.statefulMetrics == null || this.statefulMetrics.isEmpty()) {
            if (this.model != null) {
                statefulMetrics = new ArrayList<>();
                //TODO this.model.metrics.forEach(metric -> statefulMetrics.add(metric));
            } else {
                statefulMetrics = Collections.EMPTY_LIST;
            }
        }

        if (this.progbar == null) {
            this.progbar = new ProgressBar(
                    this.target,
                    this.verbose,
                    mode.steps.toString(),
                    this.statefulMetrics
            );
        }
    }

    private void finalizeProgbar(Map<String, Number> logs) {
        if (this.target == null) {
            this.target = this.seen;
            this.progbar.setTarget(this.seen);
        }
        logs = logs == null ? Collections.EMPTY_MAP : logs;
        this.progbar.update(this.seen, new ArrayList<>(logs.entrySet()), true);
    }
}
