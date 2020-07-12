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
import java.util.Map;
import java.util.function.BiFunction;
import java.util.logging.Level;
import java.util.logging.Logger;
import static javafx.scene.input.KeyCode.K;

/**
 * Callback to Reduce learning rate when a metric has stopped improving.
 *
 * <p>
 * Models often benefit from reducing the learning rate by a factor of 2-10 once
 * learning stagnates. This callback monitors a quantity and if no improvement
 * is seen for a 'patience' number of epochs, the learning rate is reduced.
 */
public class ReduceLROnPlateau extends Callback {

    /**
     * Quantity to be monitored. Default is "val_loss".
     */
    private final String monitor;

    /**
     * factor by which the learning rate will be reduced. new_lr = lr * factor.
     * Default is 0.1.
     */
    private final float learningRateFactor;

    /**
     * Minimum change in the monitored quantity to qualify as an improvement,
     * i.e. an absolute change of less than min_delta, will count as no
     * improvement. Default is 0.0001.
     */
    private float minDelta;
    /**
     * Number of epochs with no improvement after which training will be
     * stopped. Default is 10.
     */
    private final int patience;
    /**
     * verbosity mode. Default is false.
     */
    private final boolean verbose;
    /**
     * One of {"auto", "min", "max"}. In min mode, the learning rate will be
     * reduced when the quantity monitored has stopped decreasing; in max mode
     * it will be reduced when the quantity monitored has stopped increasing; in
     * auto mode, the direction is automatically inferred from the name of the
     * monitored quantity. Default is Mode.auto.
     */
    private final Mode mode;

    /**
     * number of epochs to wait before resuming normal operation after lr has
     * been reduced. Default is 0.
     */
    private final int coolDown;

    /**
     * lower bound on the learning rate. Default is 0.
     */
    private final float minLR;

    
    private double best;
    private int coolDownCounter;
    private int wait;
    private BiFunction<Number, Number, Boolean> monitor_op;
    private boolean monitorGreater;
    
    
    /**
     * Create a ReduceLROnPlateau callback
     */
    public ReduceLROnPlateau() {
        this(null, null, "val_loss", 0.1f, 10, false, Mode.auto, 0.0001f, 0, 0);
    }

    public ReduceLROnPlateau(String monitor) {
        this(null, null, monitor, 0.1f, 10, false, Mode.auto, 0.0001f, 0, 0);
    }

    public ReduceLROnPlateau(String monitor, float learningRateFactor) {
        this(null, null, monitor, learningRateFactor, 10, false, Mode.auto, 0.0001f, 0, 0);
    }

    public ReduceLROnPlateau(String monitor, float learningRateFactor, int patience) {
        this(null, null, monitor, learningRateFactor, patience, false, Mode.auto, 0.0001f, 0, 0);
    }

    public ReduceLROnPlateau(String monitor, float learningRateFactor, int patience, float minLR) {
        this(null, null, monitor, learningRateFactor, patience, false, Mode.auto, 0.0001f, 0, minLR);
    }

    public ReduceLROnPlateau(Map<String, Object> params, Object model, String monitor,
            float learningRateFactor, int patience, boolean verbose, Mode mode,
            float minDelta, int coolDown, float minLR) {
        super(params, model);
        assert learningRateFactor < 1.0 : "ReduceLROnPlateau ' 'does not support a factor >= 1.0";
        this.monitor = monitor;
        this.learningRateFactor = learningRateFactor;
        this.minDelta = Math.abs(minDelta);
        this.patience = patience;
        this.verbose = verbose;
        this.mode = mode;
        this.coolDown = coolDown;
        this.minLR = minLR;
        reset();
        
    }
    
    private void reset() {
        switch (mode) {
            case min:
                this.monitor_op = (a, b) -> a.doubleValue() < b.doubleValue();
                this.minDelta *= -1;
                this.best = Double.MAX_VALUE;
                break;
            case max:
                this.monitor_op = (a, b) -> a.doubleValue() > b.doubleValue();
                monitorGreater = true;
                this.best = Double.MIN_VALUE;
                break;
            default:
                if (this.monitor.equals("acc")) {
                    this.monitor_op = (a, b) -> a.doubleValue() > b.doubleValue();
                    monitorGreater = true;
                    //this.minDelta *= 1;
                } else {
                    this.monitor_op = (a, b) -> a.doubleValue() < b.doubleValue();
                    this.minDelta *= -1;
                }
                break;
        }
        this.coolDownCounter = 0;
        this.wait = 0;
        
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public void onTrainBegin(Map<String, Number> logs) {
        this.reset();
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public void onEpochEnd(int epoch, Map<String, Number> logs) {
        Number current = getMonitorValue(logs, this.monitor);
        if (current == null) {
            Logger.getLogger(this.getClass().getName()).warning(
                () -> String.format(
                        "Reduce LR on plateau conditioned on metric `%s` is not available. Available metrics are: %s", 
                        this.monitor, Arrays.toString(logs.keySet().toArray()))
            );
            return;
        }
        if (this.monitor_op.apply(current.doubleValue(), this.best)) {
            this.best = current.doubleValue();
            this.wait = 0;
        } else {
            if(coolDownCounter > 0){
                this.coolDownCounter--;
                this.wait = 0;
            }
            if (this.monitor_op.apply(current.doubleValue() - this.minDelta, this.best)) {
                this.best = current.doubleValue();
                this.wait = 0;
            }else {
                this.wait++;
                if(this.wait > this.patience) {
                    /** TODO
                    float oldLR = K.get_value(this.model.optimizer.lr);
                    if(oldLR > this.minLR) {
                        float newLR = oldLR * this.learningRateFactor;
                        newLR = Math.max(newLR, this.minLR);
                        K.setValue(this.model.optimizer.lr, newLR);
                        if(verbose) {
                            System.out.printf(
                                "\nEpoch %05d: ReduceLROnPlateau reducing learning rate to %f.",
                                epoch+1, newLR);
                        }
                        this.coolDownCounter = this.coolDown;
                        this.wait = 0;
                        
                    }
                    * ****/
                } 
            }

        }
    }

}
