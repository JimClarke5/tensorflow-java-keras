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
 * Stop training when a monitored metric has stopped improving.
 */
public class EarlyStopping extends Callback {

    private int wait;
    private int stoppedEpoch;


    /**
     * Quantity to be monitored. Default is "val_loss".
     */
    private String monitor;
    /**
     * Minimum change in the monitored quantity to qualify as an improvement,
     * i.e. an absolute change of less than min_delta, will count as no
     * improvement. Default is 0.
     */
    private double minDelta;
    /**
     * Number of epochs with no improvement after which training will be
     * stopped. Default is 0.
     */
    private int patience;
    /**
     * verbosity mode. Default is false.
     */
    private boolean verbose;
    /**
     * One of {"auto", "min", "max"}. In min mode, training will stop when the
     * quantity monitored has stopped decreasing; in max mode it will stop when
     * the quantity monitored has stopped increasing; in auto mode, the
     * direction is automatically inferred from the name of the monitored
     * quantity. Default is Mode.auto.
     */
    private Mode mode;
    /**
     * Baseline value for the monitored quantity. Training will stop if the
     * model doesn't show improvement over the baseline. Default is null,
     */
    private Double baseline;
    /**
     * Flag indicating whether to restore model weights from the epoch with the
     * best value of the monitored quantity. If false (default), the model
     * weights obtained at the last step of training are used.
     */
    private boolean restoreBestWeights;

    private double best;
    private boolean monitorGreater;

    private BiFunction<Number, Number, Boolean> monitor_op;

    /**
     * Create an EarlyStopping Callback
     */
    public EarlyStopping() {
        this(null, null, "val_loss", 0.0, 0, false, Mode.auto, null, false);
    }

    /**
     * Create an EarlyStopping Callback
     *
     * @param params Training parameters
     */
    public EarlyStopping(Map<String, Object> params) {
        this(params, null, "val_loss", 0.0, 0, false, Mode.auto, null, false);
    }

    /**
     * Create an EarlyStopping Callback
     *
     * @param monitor Quantity to be monitored.
     */
    public EarlyStopping(String monitor) {
        this(null, null, monitor, 0.0, 0, false, Mode.auto, null, false);
    }

    /**
     * Create an EarlyStopping Callback
     *
     * @param params Training parameters
     * @param model Reference of the model being trained.
     * @param monitor Quantity to be monitored.
     * @param minDelta Minimum change in the monitored quantity to qualify as an
     * improvement, i.e. an absolute change of less than min_delta, will count
     * as no improvement.
     * @param patience Number of epochs with no improvement after which training
     * will be stopped.
     * @param verbose verbosity mode.
     * @param mode In min mode, training will stop when the quantity monitored
     * has stopped decreasing; in max mode it will stop when the quantity
     * monitored has stopped increasing; in auto mode, the direction is
     * automatically inferred from the name of the monitored quantity.
     * @param baseline Baseline value for the monitored quantity. Training will
     * stop if the model doesn't show improvement over the baseline.
     * @param restoreBestWeights Whether to restore model weights from the epoch
     * with the best value of the monitored quantity. If false, the model
     * weights obtained at the last step of training are used.
     */
    public EarlyStopping(Map<String, Object> params, Object model, String monitor,
            double minDelta, int patience, boolean verbose, Mode mode,
            Double baseline, boolean restoreBestWeights) {
        super(params, model);
        this.monitor = monitor;
        this.minDelta = Math.abs(minDelta);
        this.patience = patience;
        this.verbose = verbose;
        this.mode = mode;
        this.baseline = baseline;
        this.restoreBestWeights = restoreBestWeights;

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
                    this.best = Double.MAX_VALUE;
                } else {
                    this.monitor_op = (a, b) -> a.doubleValue() < b.doubleValue();
                    this.minDelta *= -1;
                    this.best = Double.MIN_VALUE;
                }
                break;
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void onTrainBegin(Map<String, Number> logs) {
        this.wait = 0;
        this.stoppedEpoch = 0;
        this.best = this.baseline != null ? this.baseline
                : this.monitorGreater ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void onEpochEnd(int epoch, Map<String, Number> logs) {
        Number current = getMonitorValue(logs, this.monitor);
        if (current == null) {
            return;
        }
        if (this.monitor_op.apply(current.doubleValue() - this.minDelta, this.best)) {
            this.best = current.doubleValue();
            this.wait = 0;
            if (this.restoreBestWeights) {
                // TODO this.bestWeights = this.model.getWeights();
            }
        } else {
            this.wait++;
            if (this.wait > this.patience) {
                this.stoppedEpoch = epoch;
                //TODO this.model.stopTraining();
                if (this.restoreBestWeights) {
                    if (this.verbose) {
                        Logger.getLogger(EarlyStopping.class.getName()).log(Level.INFO,
                                "Restoring model weights from the end of the best epoch.");
                    }
                    // TODO this.model.setWeights(this.bestWeights)
                }
            }

        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void onTrainEnd(Map<String, Number> logs) {
        if (this.stoppedEpoch > 0 && this.verbose) {
            Logger.getLogger(EarlyStopping.class.getName()).log(Level.INFO,
                    String.format("Epoch %05d: early stopping: ", this.stoppedEpoch + 1));
        }
    }

    
}
