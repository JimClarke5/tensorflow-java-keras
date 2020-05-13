/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the );
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an  BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=======================================================================*/
package org.tensorflow.keras.optimizers;

import java.util.Map;
import org.tensorflow.Graph;
import static org.tensorflow.keras.optimizers.OptimizerInterface.config;

/**
 * AdaGrad Optimizer that implements the AdaGrad algorithm.
 * Adagrad is an optimizer with parameter-specific learning rates,
 * which are adapted relative to how frequently a parameter gets
 * updated during training. The more updates a parameter receives,
 * the smaller the updates.
 *
 * @author Jim Clarke
 */
public class AdaGrad extends org.tensorflow.framework.optimizers.AdaGrad implements OptimizerInterface  {

    public static final String LEARNING_RATE_KEY = "learning_rate";
    public static final String INITIAL_ACCUM_KEY = "accumulator";
    
    
    public static final float  LEARNING_RATE_DEFAULT = 0.001F;
    public static final float  INITIAL_ACCUM__DEFAULT = 0.1f;
  
    
    
    // TODO is this still necessary?
    private String[] allowed_options = {"clipnorm", "clipvalue", "lr", "decay"};
    

     /**
     * create an Adadelta Optimizer with name="Adagrad", learningRate=0.001F, and initial accumulator=0.1
     * @param graph the tensorflow graph
     */
    public AdaGrad(Graph graph) {
        this(graph, LEARNING_RATE_DEFAULT, INITIAL_ACCUM__DEFAULT);
    }
    
      /**
     * create an Adadelta Optimizer with learningRate=0.001F, and initial accumulator=0.1
     * @param graph the tensorflow graph
     * @param name the name of the Optimizer, defaults to "Adagrad"
     */
    public AdaGrad(Graph graph, String name) {
        this(graph, name, LEARNING_RATE_DEFAULT, INITIAL_ACCUM__DEFAULT);
    }
    
    
    /**
     * create an Adadelta Optimizer with  initial accumulator=0.1
     * @param graph the tensorflow graph
     * @param learningRate The learning rate. Defaults to 0.001.
     */
    public AdaGrad(Graph graph, float learningRate) {
        this(graph, learningRate, INITIAL_ACCUM__DEFAULT);
    }
    
     /**
     * create an Adadelta Optimizer
     * @param graph the tensorflow graph
     * @param name the name of the Optimizer, defaults to "Adagrad"
     * @param learningRate The learning rate. Defaults to 0.01.
     */
    public AdaGrad(Graph graph, String name, float learningRate) {
        this(graph, name, learningRate, INITIAL_ACCUM__DEFAULT);
    }
    
    /**
     * create an Adadelta Optimizer
     * @param graph the tensorflow graph
     * @param learningRate The learning rate
     * @param initialAccumulatorValue initial accumulator value
     */
    public AdaGrad(Graph graph,float learningRate, float initialAccumulatorValue) {
        super(graph, learningRate, initialAccumulatorValue);
        initConfig(learningRate, initialAccumulatorValue);
    }
    
    /**
     * create an Adadelta Optimizer
     * @param graph the tensorflow graph
     * @param name the name of the Optimizer, defaults to "Adagrad"
     * @param learningRate The learning rate
     * @param initialAccumulatorValue initial accumulator value, must be >= 0.
     */
    public AdaGrad(Graph graph, String name, float learningRate, float initialAccumulatorValue) {
        super(graph, name, learningRate, initialAccumulatorValue);
        assert initialAccumulatorValue >= 0.0F : "initial_accumulator_value must be non-negative: " + initialAccumulatorValue;
        initConfig(learningRate, initialAccumulatorValue);
    }
    
    
    
    /* TODO - do we need to do this to be compatible with keras python? */
    /**
     * create an AdaGrad Optimizer from a config object
     *
     * @param graph the tensorflow graph
     * @param config a config object to initialize, , the config object has keys for 
     * "name", "learning_rate" and "accumulator". If a key is missing 
     * the default value is used.
     */
    public static AdaGrad fromConfig(Graph graph, Map<String, Object> config) {
        return create(graph, config);
    }

    /**
     * create an Adadelta Optimizer from a config object
     *
     * @param graph the tensorflow graph
     * @param config a config object to initialize, the config object has keys for 
     * "name", "learning_rate" and "accumulator". If a key is missing 
     * the default value is used.
     */
    public static AdaGrad create(Graph graph, Map<String, Object> config) {
        String name = (String)config.get(NAME_KEY);
        float learningRate = (float)config.getOrDefault(LEARNING_RATE_KEY, LEARNING_RATE_DEFAULT);
        float initialAccumulatorValue = (float)config.getOrDefault(INITIAL_ACCUM_KEY, INITIAL_ACCUM__DEFAULT);
        if(name != null)
            return new AdaGrad(graph, name, learningRate, initialAccumulatorValue);
        else
            return new AdaGrad(graph, learningRate, initialAccumulatorValue);
        
    }
    
    /**
     * Initialize the configuration
     * @param learningRate
     * @param initialAccumulatorValue
     */
    private void initConfig(float learningRate, float initialAccumulatorValue) {
        config.put(NAME_KEY, this.getOptimizerName());
        config.put(LEARNING_RATE_KEY, learningRate);
        config.put(INITIAL_ACCUM_KEY, initialAccumulatorValue);
    }
    
    
    
    //TODO ??
    //variables()
    //set_weights
    //get_weights
    //add_weight
    //get_updates
    //get_slot_names
    //get_gradients
    //add_slot => createSlot
   
}
