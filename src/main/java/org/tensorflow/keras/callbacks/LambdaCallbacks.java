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

import java.util.Map;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

/**
 * Callback for creating simple, custom callbacks on-the-fly.
 * <p>
 * Example: 
 * <code>
 * LambdaCallbacks batchPrintCallback = new LambdaCallbacks();
 * c.setOnTrainBatchBegin((batch, logs)-> System.out.println("Batch: " + batch + " started");
 * </code>
 * 
 * @author jbclarke
 */
public class LambdaCallbacks extends Callback {

    /**
     * Called at the beginning of every epoch.
     * expect two positional arguments: `epoch`, `logs`
     */
    private BiConsumer<Integer, Map<String, Number>> onEpochBegin;

    /**
     * Called at the end of every epoch.
     * expect two positional arguments: `epoch`, `logs`
     */
    private BiConsumer<Integer, Map<String, Number>> onEpochEnd;

    /**
     * Called at the beginning of every batch.
     * expect two positional arguments: `batch`, `logs`
     */
    private BiConsumer<Integer, Map<String, Number>> onTrainBatchBegin;

    /**
     * called at the end of every batch.
     * expect two positional arguments: `batch`, `logs`
     */
    private BiConsumer<Integer, Map<String, Number>> onTrainBatchEnd;

    /**
     * called at the beginning of model training.
     * expect one positional argument: `logs`
     */
    private Consumer<Map<String, Number>> onTrainBegin;

    /**
     * called at the end of model training.
     * expect one positional argument: `logs`
     */
    private Consumer<Map<String, Number>> onTrainEnd;
    

    /**
     * Create a LambdaCallbacks callback
     */
    public LambdaCallbacks() {
        super();
    }

    /**
     * Create a LambdaCallbacks callback
     *
     * @param params Training parameters
     */
    public LambdaCallbacks(Map<String, Object> params) {
        super(params);
    }

    /**
     * Create a LambdaCallbacks callback
     *
     * @param params Training parameters
     * @param model Reference of the model being trained.
     */
    public LambdaCallbacks(Map<String, Object> params, Object model) {
        super(params, model);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void onEpochBegin(int epoch, Map<String, Number> logs) {
        if(this.onEpochBegin != null) {
            this.onEpochBegin.accept(epoch, logs);
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void onEpochEnd(int epoch, Map<String, Number> logs) {
        if(this.onEpochEnd != null) {
            this.onEpochEnd.accept(epoch, logs);
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void onTrainBatchBegin(int batch, Map<String, Number> logs) {
        if(this.onTrainBatchBegin != null) {
            this.onTrainBatchBegin.accept(batch, logs);
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void onTrainBatchEnd(int batch, Map<String, Number> logs) {
        if(this.onTrainBatchEnd != null) {
            this.onTrainBatchEnd.accept(batch, logs);
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void onTrainBegin(Map<String, Number> logs) {
        if(this.onTrainBegin != null) {
            this.onTrainBegin.accept(logs);
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void onTrainEnd(Map<String, Number> logs) {
        if(this.onTrainEnd != null) {
            this.onTrainEnd.accept(logs);
        }
    }

    
    /**
     * @return the onEpochBegin
     */
    public BiConsumer<Integer, Map<String, Number>> getOnEpochBegin() {
        return onEpochBegin;
    }

    /**
     * @param onEpochBegin the onEpochBegin to set
     */
    public void setOnEpochBegin(BiConsumer<Integer, Map<String, Number>> onEpochBegin) {
        this.onEpochBegin = onEpochBegin;
    }

    /**
     * @return the onEpochEnd
     */
    public BiConsumer<Integer, Map<String, Number>> getOnEpochEnd() {
        return onEpochEnd;
    }

    /**
     * @param onEpochEnd the onEpochEnd to set
     */
    public void setOnEpochEnd(BiConsumer<Integer, Map<String, Number>> onEpochEnd) {
        this.onEpochEnd = onEpochEnd;
    }

    /**
     * @return the onBatchBegin
     */
    public BiConsumer<Integer, Map<String, Number>> getOnBatchchBegin() {
        return onTrainBatchBegin;
    }

    /**
     * @param onBatchBegin the onBatchBegin to set
     */
    public void setOnTrainBatchBegin(BiConsumer<Integer, Map<String, Number>> onTrainBatchBegin) {
        this.onTrainBatchBegin = onTrainBatchBegin;
    }

    /**
     * @return the onBatchEnd
     */
    public BiConsumer<Integer, Map<String, Number>> getOnTrainBatchchEnd() {
        return onTrainBatchEnd;
    }

    /**
     * @param onTrainBatchEnd the onTrainBatchEnd to set
     */
    public void setOnTrainBatchchEnd(BiConsumer<Integer, Map<String, Number>> onTrainBatchEnd) {
        this.onTrainBatchEnd = onTrainBatchEnd;
    }

    /**
     * @return the onTrainBegin
     */
    public Consumer<Map<String, Number>> getOnTrainBegin() {
        return onTrainBegin;
    }

    /**
     * @param onTrainBegin the onTrainBegin to set
     */
    public void setOnTrainBegin(Consumer<Map<String, Number>> onTrainBegin) {
        this.onTrainBegin = onTrainBegin;
    }

    /**
     * @return the onTrainEnd
     */
    public Consumer<Map<String, Number>> getOnTrainEnd() {
        return onTrainEnd;
    }

    /**
     * @param onTrainEnd the onTrainEnd to set
     */
    public void setOnTrainEnd(Consumer<Map<String, Number>> onTrainEnd) {
        this.onTrainEnd = onTrainEnd;
    }

}
