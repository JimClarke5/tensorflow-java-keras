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

import java.io.File;
import java.util.Collections;
import java.util.Map;
import java.util.function.BiFunction;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.tensorflow.keras.utils.PlaceholderStringFormat;

/**
 * Callback to save the Keras model or model weights at some frequency.
 */
public class ModelCheckpoint extends Callback {

    public enum Mode {
        auto, min, max
    };
    public static final int EPOCH = -1;

    private boolean monitorGreater;
    private int currentEpoch;
    private String filepath;
    private File filePathFile;
    private String monitor;
    private boolean verbose;
    private boolean saveBestOnly;
    private boolean saveWeightsOnly;
    private Mode mode;
    private int saveFreq;

    private int epochsSinceLastSave;
    private int batchesSeenSinceLastSaving;
    private double best;
    private int period;

    private BiFunction<Number, Number, Boolean> monitor_op;

    /**
     * Create a ModelCheckpoint Callback
     *
     * @param filename
     */
    public ModelCheckpoint(String filename) {
        this(null, null, filename);
    }

    /**
     * Create a ModelCheckpoint Callback
     *
     * @param params Training parameters
     * @param filename
     */
    public ModelCheckpoint(Map<String, Object> params, String filename) {
        this(params, null, filename);
    }

    /**
     * Create a ModelCheckpoint Callback
     *
     * @param params Training parameters
     * @param model Reference of the model being trained.
     * @param filename
     */
    public ModelCheckpoint(Map<String, Object> params, Object model, String filename) {
        this(params, model, filename, "val_loss", false, false, false, Mode.auto, EPOCH);
    }

    /**
     * Create a ModelCheckpoint Callback
     *
     * @param params Training parameters
     * @param model Reference of the model being trained.
     * @param filepath string, path to save the model file. filepath can contain
     * named formatting options, which will be filled the value of epoch and
     * keys in logs (passed in on_epoch_end). For example: if filepath is
     * weights.{epoch:02d}-{val_loss:.2f}.hdf5, then the model checkpoints will
     * be saved with the epoch number and the validation loss in the filename.
     * @param monitor quantity to monitor.
     * @param verbose verbosity mode
     * @param saveBestOnly if saveBestOnly=true, the latest best model according
     * to the quantity monitored will not be overwritten. If filepath doesn't
     * contain formatting options like {epoch} then filepath will be overwritten
     * by each new better model.
     * @param saveWeightsOnly f True, then only the model's weights will be
     * saved (model.save_weights(filepath)), else the full model is saved
     * (model.save(filepath)).
     * @param mode If saveBestOnly=true, the decision to overwrite the current
     * save file is made based on either the maximization or the minimization of
     * the monitored quantity. For val_acc, this should be max, for val_loss
     * this should be min, etc. In auto mode, the direction is automatically
     * inferred from the name of the monitored quantity.
     * @param saveFreq When using EPOCH, the callback saves the model after each
     * epoch. When using integer >= 0, the callback saves the model at end of
     * this many batches. Note that if the saving isn't aligned to epochs, the
     * monitored metric may potentially be less reliable (it could reflect as
     * little as 1 batch, since the metrics get reset every epoch). Defaults to
     * EPOCH.
     */
    public ModelCheckpoint(Map<String, Object> params, Object model,
            String filepath, String monitor, boolean verbose, boolean saveBestOnly,
            boolean saveWeightsOnly, Mode mode, int saveFreq) {
        super(params, model);
        this.filepath = filepath;
        this.filePathFile = new File(filepath);
        this.monitor = monitor;
        this.verbose = verbose;
        this.saveBestOnly = saveBestOnly;
        this.saveWeightsOnly = saveWeightsOnly;
        this.mode = mode;
        this.saveFreq = saveFreq;

        switch (mode) {
            case min:
                this.monitor_op = (a, b) -> a.doubleValue() < b.doubleValue();
                this.best = Double.POSITIVE_INFINITY;
                break;
            case max:
                this.monitor_op = (a, b) -> a.doubleValue() > b.doubleValue();
                monitorGreater = true;
                this.best = Double.NEGATIVE_INFINITY;
                //this.minDelta *= 1;
                break;
            default:
                if (this.monitor.equals("acc") || this.monitor.startsWith("fmeasure")) {
                    this.monitor_op = (a, b) -> a.doubleValue() > b.doubleValue();
                    monitorGreater = true;
                    this.best = Double.NEGATIVE_INFINITY;
                } else {
                    this.monitor_op = (a, b) -> a.doubleValue() < b.doubleValue();
                    this.best = Double.POSITIVE_INFINITY;
                }
                break;
        }

        // TODO Workers
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void setModel(Object model) {
        super.setModel(model);
        /**
         * TODO if(!this.saveWeightsOnly && !model.isGraph() && !(model
         * instanceof Sequential)) { this.saveWeightsOnly = true; } **
         */
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void onTrainBegin(Map<String, Number> logs) {
        // TODO multi-worker recover

    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void onTrainEnd(Map<String, Number> logs) {
        // TODO multi-worker mode
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void onTrainBatchEnd(int batch, Map<String, Number> logs) {
        if (this.saveFreq != EPOCH) {
            logs = logs == null ? Collections.EMPTY_MAP : logs;
            this.batchesSeenSinceLastSaving++;
            if (this.batchesSeenSinceLastSaving >= this.saveFreq) {
                this.saveModel(this.currentEpoch, logs);
                this.batchesSeenSinceLastSaving = 0;
            }
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void onEpochBegin(int epoch, Map<String, Number> logs) {
        this.currentEpoch = epoch;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void onEpochEnd(int epoch, Map<String, Number> logs) {
        this.epochsSinceLastSave++;
        if (this.saveFreq == EPOCH) {
            //TODO multi-worker mode
            this.saveModel(epoch, logs);

        }

    }

    private void saveModel(int epoch, Map<String, Number> logs) {
        logs = logs == null ? Collections.EMPTY_MAP : logs;
        if (this.saveFreq != EPOCH || this.epochsSinceLastSave >= this.period) {
            this.epochsSinceLastSave = 0;
            String checkpoint = getFilePath(epoch, logs);

            if (this.saveBestOnly) {
                Number currentVal = logs.get(this.monitor);
                if (currentVal == null) {
                    Logger.getLogger(ModelCheckpoint.class.getName()).log(Level.WARNING,
                            String.format("can save best model only with %s available, skipping.", this.monitor));
                } else {
                    if (this.monitor_op.apply(currentVal.doubleValue(), this.best)) {
                        if (this.verbose) {
                            System.out.printf(
                                    "\nEpoch %05d: %s improved from %01.5f to %01.5f, saving model to %s",
                                    epoch + 1, this.monitor, this.best, currentVal.doubleValue(), checkpoint);
                        }
                        this.best = currentVal.doubleValue();

                        /**
                         * TODO if(this.saveWeightsOnly) {
                         * this.model.save_weights(filepath, true); }else {
                         * this.model.save(filepath, true); } ***
                         */
                    } else {
                        if (this.verbose) {
                            System.out.printf("\npoch %05d: %s did not improve from %01.5f",
                                    epoch + 1, this.monitor, this.best);
                        }
                    }
                }
            } else {
                if (this.verbose) {
                    System.out.printf("\nEpoch %05d: saving model to %s", epoch + 1, filepath);
                }
                /**
                 * TODO if(this.saveWeightsOnly) {
                 * this.model.save_weights(filepath, true); }else {
                 * this.model.save(filepath, true); } ***
                 */
            }
            //TOOD multi-worker 
            this.maybeRemoveFile();

        }
    }

    private String getFilePath(int epoch, Map<String, Number> logs) {
        return PlaceholderStringFormat.convertFilePath(this.filepath, epoch, logs);
    }

    private void maybeRemoveFile() {
        // TODO multi-worker
    }

    private String getMostRecentlyModifiedFileMatchingPattern(String pattern) {
        // TODO
        return "";
    }

}
