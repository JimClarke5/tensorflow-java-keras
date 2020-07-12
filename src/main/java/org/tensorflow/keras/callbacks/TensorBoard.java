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
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.util.HashMap;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.tensorflow.op.Ops;

/**
 * Enable visualizations for TensorBoard.
 * <p>
 * TensorBoard is a visualization tool provided with TensorFlow.
 * <p>
 * This callback logs events for TensorBoard, including:
 * <UL>
 * <LI>Metrics summary plots
 * <LI>Training graph visualization
 * <LI>Activation histograms
 * <LI>Sampled profiling
 * </UL>
 * If you have installed TensorFlow with pip, you should be able to launch
 * TensorBoard from the command line:
 * <code> tensorboard --logdir=path_to_your_logs</code> You can find more
 * information about TensorBoard
 * <a href="https://www.tensorflow.org/get_started/summaries_and_tensorboard">[here]</a>
 */
public class TensorBoard extends Callback {

    public static final int EPOCH = -1;
    public static final int BATCH = -2;

    String logDir;
    int histogramFreq;
    boolean writeGraph;
    boolean writeImages;
    int updateFreq;
    // TODO update when Eager Mode is supported
    int[] profileBatch;
    int[] embeddingsFreq;
    Map<String, String> embeddingsMetadata;
    
    private int samplesSeen;
    private int samplesSeenAtLastWrite;
    private int currentBatch;
    private String trainRunName = "train";
    private String validationRunName = "validation";
    private Map<String, Writer>  writers = new HashMap<>();
    private boolean isTracing;
    private int startBatch;
    private int stopBatch;
    private File logWriteDir; 
    
    /**
     * Create a TensorBoard callback
     *
     * @param logDir
     */
    public TensorBoard(String logDir) {
        this(null, null, logDir, 0, true, false, EPOCH,
                new int[]{2}, new int[]{0}, null);
    }

    /**
     * Create a TensorBoard callback
     *
     * @param params Training parameters
     * @param model Reference of the model being trained.
     * @param logDir the path of the directory where to save the log files to be
     * parsed by TensorBoard.
     * @param histogramFreq frequency (in epochs) at which to compute activation
     * and weight histograms for the layers of the model. If set to 0,
     * histograms won't be computed.
     * @param writeGraph whether to visualize the graph in TensorBoard. The log
     * file can become quite large when write_graph is set to true.
     * @param writeImages whether to write model weights to visualize as image
     * in TensorBoard.
     * @param updateFreq When using `'batch'`, writes the losses and metrics to
     * TensorBoard after each batch. The same applies for `'epoch'`. If using an
     * integer, let's say `1000`, the callback will write the metrics and losses
     * to TensorBoard every 1000 batches. Note that writing too frequently to
     * TensorBoard can slow down your training.
     * @param profileBatch Profile the batch(es) to sample compute
     * characteristics. profile_batch must be a non-negative integer or a comma
     * separated string of pair of positive integers. A pair of positive
     * integers signify a range of batches to profile. By default, it will
     * profile the second batch. Set profile_batch=0 to disable profiling. Must
     * run in TensorFlow eager mode.
     * @param embeddingsFreq frequency (in epochs) at which embedding layers
     * will be visualized. If set to 0, embeddings won't be visualized.
     * @param embeddingsMetadata a dictionary which maps layer name to a file
     * name in which metadata for this embedding layer is saved.
     */
    public TensorBoard(Map<String, Object> params, Object model,
            String logDir, int histogramFreq, boolean writeGraph,
            boolean writeImages,
            int updateFreq,
            // TODO update when Eager Mode is supported
            int[] profileBatch,
            int[] embeddingsFreq,
            Map<String, String> embeddingsMetadata) {
        super(params, model);
        assert profileBatch.length <= 2 : "profile_batch must either be one integer or a pair of two integers";
        this.logDir = logDir;
        this.histogramFreq = histogramFreq;
        this.writeGraph = writeGraph;
        this.writeImages = writeImages;
        this.updateFreq = updateFreq == BATCH? 1 : updateFreq;
        this.profileBatch = profileBatch;
        this.embeddingsFreq = embeddingsFreq;
        this.embeddingsMetadata = embeddingsMetadata;
        
        if(profileBatch.length == 1){
            this.startBatch = profileBatch[0];
            this.stopBatch = profileBatch[0];
        }else if(profileBatch.length == 2) {
            this.startBatch = profileBatch[0];
            this.stopBatch = profileBatch[1];
        }
        assert this.startBatch >= 0 : "Start Batch must be greater than or equal to 0.";
        assert this.stopBatch >= 0 : "Start Batch must be greater than or equal to 0.";
        assert this.startBatch <= this.stopBatch : "Start Batch must be greater than or equal to Stop Batch";
        if(startBatch > 0) {
            // TODO profiler.warmup();
        }
        
        this.logWriteDir = new File(this.logDir);
        

    }
     // TODO replace with Model
    /**
    * {@inheritDoc}
    */
    @Override
    public void setModel(Object model) {
        super.setModel(model);
        
        // TODO
        /**
        summary_state = summary_ops_v2._summary_state  # pylint: disable=protected-access
        self._prev_summary_recording = summary_state.is_recording
        self._prev_summary_writer = summary_state.writer
        self._prev_summary_step = summary_state.step
        * **/
    }
    
    private Writer getWriter(String name) {
        Writer writer = writers.get(name);
        if(writer == null) {
            try {
                writer =  new FileWriter(new File(this.logWriteDir, name));
                writers.put(name, writer);
            } catch (IOException ex) {
                Logger.getLogger(TensorBoard.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        return writer;
    }
    
    // TODO Much to do yet.

}
