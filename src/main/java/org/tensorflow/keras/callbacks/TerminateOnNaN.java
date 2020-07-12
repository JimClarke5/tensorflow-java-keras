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

/**
 * Callback that terminates training when a NaN loss is encountered.
 */
public class TerminateOnNaN extends Callback {

    /**
     * Create a TerminateOnNaN Callback
     */
    public TerminateOnNaN() {
        this(null, null);
    }

    /**
     * Create a TerminateOnNaN Callback
     *
     * @param params Training parameters
     */
    public TerminateOnNaN(Map<String, Object> params) {
        this(params, null);
    }

    /**
     * Create a TerminateOnNaN Callback
     *
     * @param params Training parameters
     * @param model Reference of the model being trained.
     */
    public TerminateOnNaN(Map<String, Object> params, Object model) {
        this.params = params;
        this.model = model;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void onTrainBatchEnd(int batch, Map<String, Number> logs) {
        logs = logs == null ? Collections.EMPTY_MAP : logs;
        Number loss = loss = logs.get("loss");
        if (loss != null) {
            if (loss.doubleValue() == Double.NaN
                    || loss.doubleValue() == Double.POSITIVE_INFINITY
                    || loss.doubleValue() == Double.NEGATIVE_INFINITY) {
                System.out.printf("Batch %d: Invalid loss, terminating training", batch);
                // TODO this.model.setStopTraining(true);
            }
        }

    }
}
