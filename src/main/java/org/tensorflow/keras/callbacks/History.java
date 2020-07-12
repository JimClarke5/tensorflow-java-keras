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
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Callback that records events into a History object.
 */
public class History extends Callback {

    private final Map<String, List<Number>> history = new HashMap<>();
    private final List<Integer> epoch = new ArrayList<>();

    /**
     * Create a History Callback
     */
    protected History() {
        this(null, null);
    }

    /**
     * Create a History Callback
     *
     * @param params Training parameters
     */
    protected History(Map<String, Object> params) {
        this(params, null);
    }

    /**
     * Create a History Callback
     *
     * @param params Training parameters
     * @param model Reference of the model being trained.
     */
    protected History(Map<String, Object> params, Object model) {
        super(params, model);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void onTrainBegin(Map<String, Number> logs) {
        epoch.clear();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void onEpochEnd(int epoch, Map<String, Number> logs) {
        logs = logs == null ? Collections.EMPTY_MAP : logs;
        this.epoch.add(epoch);

        final Map<String, Number> finalLogs = logs;
        logs.keySet().forEach(key -> {
            List<Number> historyList = this.history.get(key);
            if (historyList == null) {
                historyList = new ArrayList<>();
                this.history.put(key, historyList);
            }
            historyList.add(finalLogs.get(key));
        });

        // TODO
        //this.model.history = this;
    }
}
