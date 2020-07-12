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

import java.util.Collections;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Abstract base class used to build new callbacks.
 */
public abstract class Callback {

    protected Object model; // TODO Change to Model when it is ready
    protected Map<String, Object> params;

    /**
     * Create a Callback
     */
    protected Callback() {
        this(null, null);
    }

    /**
     * Create a Callback
     *
     * @param params Training parameters
     */
    protected Callback(Map<String, Object> params) {
        this(params, null);
    }

    /**
     * Create a Callback
     *
     * @param params Training parameters
     * @param model Reference of the model being trained.
     */
    protected Callback(Map<String, Object> params, Object model) {
        this.params = params;
        this.model = model;
    }

    /**
     * Called at the start of an epoch. This method should only be called during
     * TRAIN mode. This method is empty. Extend this class to handle this event.
     *
     * @param epoch index of epoch.
     * @param logs metric results
     */
    public void onEpochBegin(int epoch, Map<String, Number> logs) {
    }

    /**
     * Called at the end of an epoch.This method should only be called during
     * TRAIN mode. This method is empty. Extend this class to handle this event.
     *
     * @param epoch index of epoch.
     * @param logs metric results for this training epoch, and for the
     * validation epoch if validation is performed. Validation result keys are
     * prefixed with `val_`.
     */
    public void onEpochEnd(int epoch, Map<String, Number> logs) {
    }

    /**
     * Called at the beginning of a training batch in `fit` methods. This method
     * is empty. Extend this class to handle this event.
     *
     * @param batch the batch index
     * @param logs Has keys `batch` and `size` representing the current batch
     * number and the size of the batch.
     */
    public void onTrainBatchBegin(int batch, Map<String, Number> logs) {
    }

    /**
     * Called at the end of a training batch in `fit` methods. This method is
     * empty. Extend this class to handle this event.
     *
     * @param batch index of batch within the current epoch.
     * @param logs Metric results for this batch.
     */
    public void onTrainBatchEnd(int batch, Map<String, Number> logs) {
    }

    /**
     * Called at the beginning of training. This method is empty. Extend this
     * class to handle this event.
     *
     * @param logs metric results
     */
    public void onTrainBegin(Map<String, Number> logs) {
    }

    /**
     * Called at the end of training. This method is empty. Extend this class to
     * handle this event.
     *
     * @param logs metric results
     */
    public void onTrainEnd(Map<String, Number> logs) {
    }

    /**
     * Called at the beginning of a batch in `evaluate` methods. Also called at
     * the beginning of a validation batch in the `fit` methods, if validation
     * data is provided. This method is empty. Extend this class to handle this
     * event.
     *
     * @param batch the batch number
     * @param logs Has keys `batch` and `size` representing the current batch
     * number and the size of the batch.
     */
    public void onTestBatchBegin(int batch, Map<String, Number> logs) {
    }

    /**
     * Called at the end of a batch in `evaluate` methods. Also called at the
     * end of a validation batch in the `fit` methods, if validation data is
     * provided.
     *
     * This method is empty. Extend this class to handle this event.
     *
     * @param batch the batch number
     * @param logs Metric results for this batch.
     */
    public void onTestBatchEnd(int batch, Map<String, Number> logs) {
    }

    /**
     * Called at the beginning of evaluation or validation. This method is
     * empty. Extend this class to handle this event.
     *
     * @param logs metric results
     */
    public void onTestBegin(Map<String, Number> logs) {
    }

    /**
     * Called at the end of evaluation or validation. This method is empty.
     * Extend this class to handle this event.
     *
     * @param logs metric results
     */
    public void onTestEnd(Map<String, Number> logs) {
    }

    /**
     * Called at the beginning of a batch in `predict` methods. This method is
     * empty. Extend this class to handle this event.
     *
     * @param batch index of batch within the current epoch.
     * @param logs Has keys `batch` and `size` representing the current batch
     * number and the size of the batch.
     */
    public void onPredictBatchBegin(int batch, Map<String, Number> logs) {
    }

    /**
     * Called at the end of a batch in `predict` methods. This method is empty.
     * Extend this class to handle this event.
     *
     * @param batch index of batch within the current epoch.
     * @param logs Metric results for this batch.
     */
    public void onPredictBatchEnd(int batch, Map<String, Number> logs) {
    }

    /**
     * Called at the beginning of prediction. This method is empty. Extend this
     * class to handle this event.
     *
     * @param logs metric results
     */
    public void onPredictBegin(Map<String, Number> logs) {
    }

    /**
     * Called at the end of prediction. This method is empty. Extend this class
     * to handle this event.
     *
     * @param logs metric results
     */
    public void onPredictEnd(Map<String, Number> logs) {
    }
    
    protected Number getMonitorValue(Map<String, Number> logs, String monitor) {
        logs = logs == null ? Collections.EMPTY_MAP : logs;
        Number monitorValue = logs.get(monitor);
        if (monitorValue != null) {
            Logger.getLogger(EarlyStopping.class.getName()).log(Level.WARNING,
                    String.format("Early stopping conditioned on metric `%s` which is not available. Available metrics are: %s",
                            monitor, String.join(",", logs.keySet())));
        }
        return monitorValue;
    }

    /**
     * @return the model
     */
    public Object getModel() {
        return model;
    }

    /**
     * @param model the model to set
     */
    public void setModel(Object model) {
        this.model = model;
    }

    /**
     * @return the params
     */
    public Map<String, Object> getParams() {
        return params;
    }

    /**
     * @param params the params to set
     */
    public void setParams(Map<String, Object> params) {
        this.params = params;
    }
}
