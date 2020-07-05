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
import java.util.function.BiFunction;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author jbclarke
 */
public class LearningRateScheduler extends Callback {

    private final BiFunction<Integer, Float, Float> schedule;
    private final boolean verbose;

    /**
     * Create a LearningRateScheduler
     *
     * @param schedule a function that takes an epoch index as input (integer,
     * indexed from 0) and returns a new learning rate as output (double)
     */
    protected LearningRateScheduler(BiFunction<Integer, Float, Float> schedule) {
        this(null, null, schedule, false);
    }

    /**
     * Create a LearningRateScheduler
     *
     * @param params Training parameters
     * @param schedule a function that takes an epoch index as input (integer,
     * indexed from 0) and returns a new learning rate as output (double)
     */
    protected LearningRateScheduler(Map<String, Object> params, BiFunction<Integer, Float, Float> schedule) {
        this(params, null, schedule, false);
    }

    /**
     * Create a LearningRateScheduler
     *
     * @param params Training parameters
     * @param model Reference of the model being trained.
     * @param schedule a function that takes an epoch index as input (integer,
     * indexed from 0) and returns a new learning rate as output (double)
     */
    protected LearningRateScheduler(Map<String, Object> params, Object model,
            BiFunction<Integer, Float, Float> schedule) {
        this(params, model, schedule, false);
    }

    /**
     * Create a LearningRateScheduler
     *
     * @param params Training parameters
     * @param model Reference of the model being trained.
     * @param schedule a function that takes an epoch index as input (integer,
     * indexed from 0) and returns a new learning rate as output (double)
     * @param verbose if true, log messages
     */
    protected LearningRateScheduler(Map<String, Object> params, Object model,
            BiFunction<Integer, Float, Float> schedule, boolean verbose) {
        this.params = params;
        this.model = model;
        this.verbose = verbose;
        this.schedule = schedule;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void onEpochBegin(int epoch, Map<String, Number> logs) {
        float lr = 0; // TODO K.get_value(this.model.optimizer.getLearningRate());
        lr = this.schedule.apply(epoch, lr);
        //TODO this.model.optimizer.setLearningRate(lr);
        if (this.verbose) {
            Logger.getLogger(LearningRateScheduler.class.getName()).log(Level.INFO,
                    String.format("Epoch %05d: LearningRateScheduler reducing learning '\n rate to %f.",
                            epoch, lr));
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void onEpochEnd(int epoch, Map<String, Number> logs) {
        logs = logs == null ? Collections.EMPTY_MAP : logs;
        //TODO logs.put("lr", this.model.optimizer.getLearningRate());
    }

}
