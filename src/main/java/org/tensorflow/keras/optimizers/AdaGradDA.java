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

import java.util.HashMap;
import java.util.Map;
import org.tensorflow.Graph;

// TODO does this make sense to include in Keras, it's not in tensorflow.keras.
/**
 * Optimizer that implements the Adagrad Dual-Averaging algorithm.
 *
 * @author Jim Clarke
 */
public class AdaGradDA extends org.tensorflow.framework.optimizers.AdaGradDA implements OptimizerInterface {

    public static final String LEARNING_RATE_KEY = "learning_rate";
    public static final String INITIAL_ACCUM_KEY = "accumulator";
    public static final String L1STRENGTH_KEY = "l1Strength";
    public static final String L2STRENGTH_KEY = "l2Strength";

    public static final float LEARNING_RATE_DEFAULT = 0.001F; // arbitray number
    public static final float INITIAL_ACCUM__DEFAULT = 0.1f;
    public static final float L1STRENGTH_DEFAULT = 0.0F;
    public static final float L2STRENGTH_DEFAULT = 0.0F;

    private Map<String, Object> config = new HashMap<>();
    private float learningRate;

    /**
     * create an AdagradDA Optimizer with default values name="adagrad-da".
     * learning_rate=.001, initial accumulator= 0.1, l1Strength=0.0,
     * l2Strength=0.0;
     *
     * @param graph the tensorflow graph
     */
    public AdaGradDA(Graph graph) {
        this(graph, LEARNING_RATE_DEFAULT, INITIAL_ACCUM__DEFAULT, L1STRENGTH_DEFAULT, L2STRENGTH_DEFAULT);
    }

    /**
     * create an AdagradDA Optimizer with default values initial accumulator=
     * 0.1, l1Strength=0.0, l2Strength=0.0;
     *
     * @param graph the tensorflow graph
     * @param learningRate The learning rate.
     */
    public AdaGradDA(Graph graph, float learningRate) {
        this(graph, learningRate, INITIAL_ACCUM__DEFAULT, L1STRENGTH_DEFAULT, L2STRENGTH_DEFAULT);
    }

    /**
     * create an AdagradDA Optimizer with default values initial accumulator=
     * 0.1, l1Strength=0.0, l2Strength=0.0;
     *
     * @param graph the tensorflow graph
     * @param name the name of the Optimizer, defaults to "adagrad-da"
     * @param learningRate The learning rate.
     */
    public AdaGradDA(Graph graph, String name, float learningRate) {
        this(graph, name, learningRate, INITIAL_ACCUM__DEFAULT, L1STRENGTH_DEFAULT, L2STRENGTH_DEFAULT);
    }

    /**
     * create an AdagradDA Optimizer
     *
     * @param graph the tensorflow graph
     * @param learningRate the learning rate, default is 0.001
     * @param initialAccumulatorValue Starting value for the accumulators, must
     * be >= 0.0.
     * @param l1Strength L1 Regularization Strength
     * @param l2Strength L2 Regularization Strength
     */
    public AdaGradDA(Graph graph, float learningRate, float initialAccumulatorValue, float l1Strength,
            float l2Strength) {
        super(graph, learningRate, initialAccumulatorValue, l1Strength, l2Strength);
        assert initialAccumulatorValue >= 0.0F : "initial_accumulator_value must be non-negative: " + initialAccumulatorValue;
        assert l1Strength >= 0.0F : "l1Strength must be non-negative: " + l1Strength;
        assert l2Strength >= 0.0F : "l2Strength must be non-negative: " + l2Strength;
        initConfig(learningRate, initialAccumulatorValue, l1Strength, l2Strength);
    }

    /**
     * create an AdagradDA Optimizer
     *
     * @param graph the tensorflow graph
     * @param name the name of the Optimizer, defaults to "adagrad-da"
     * @param learningRate the learning rate, default is 0.001
     * @param initialAccumulatorValue Starting value for the accumulators, must
     * be positive.
     * @param l1Strength L1 Regularization Strength
     * @param l2Strength L2 Regularization Strength
     */
    public AdaGradDA(Graph graph, String name, float learningRate, float initialAccumulatorValue, float l1Strength,
            float l2Strength) {
        super(graph, name, learningRate, initialAccumulatorValue, l1Strength, l2Strength);
        assert initialAccumulatorValue >= 0.0F : "initial_accumulator_value must be non-negative: " + initialAccumulatorValue;
        assert l1Strength >= 0.0F : "l1Strength must be non-negative: " + l1Strength;
        assert l2Strength >= 0.0F : "l2Strength must be non-negative: " + l2Strength;
        initConfig(learningRate, initialAccumulatorValue, l1Strength, l2Strength);
    }

    /* TODO - do we need to do this to be compatible with keras python? */
    /**
     * create an AdaGrad Optimizer from a config object
     *
     * @param graph the tensorflow graph
     * @param config a config object to initialize, , the config object has keys
     * for "name", "learning_rate", "accumulator", "l1Strength" and
     * "l2Strength". If a key is missing the default value is used.
     * @return the new AdaGradDA Optimizer
     */
    public static AdaGradDA fromConfig(Graph graph, Map<String, Object> config) {
        return create(graph, config);
    }

    /**
     * create an AdaGradDA Optimizer from a config object
     *
     * @param graph the tensorflow graph
     * @param config a config object to initialize, the config object has keys
     * for "name", "learning_rate", "accumulator", "l1Strength" and
     * "l2Strength". If a key is missing the default value is used.
     * @return the new AdaGradDA Optimizer
     */
    public static AdaGradDA create(Graph graph, Map<String, Object> config) {
        String name = (String) config.get(NAME_KEY);
        float learningRate = (float) config.getOrDefault(LEARNING_RATE_KEY, LEARNING_RATE_DEFAULT);
        float initialAccumulatorValue = (float) config.getOrDefault(INITIAL_ACCUM_KEY, INITIAL_ACCUM__DEFAULT);
        float l1Strength = (float) config.getOrDefault(L1STRENGTH_KEY, L2STRENGTH_DEFAULT);
        float l2Strength = (float) config.getOrDefault(L2STRENGTH_KEY, L2STRENGTH_DEFAULT);
        if (name != null) {
            return new AdaGradDA(graph, name, learningRate, initialAccumulatorValue, l1Strength, l2Strength);
        } else {
            return new AdaGradDA(graph, learningRate, initialAccumulatorValue, l1Strength, l2Strength);
        }

    }

    /**
     * function that sets the config object based on which constructor is
     * called.
     *
     * @param learningRate the learning rate, default is 0.001
     * @param initialAccumulatorValue Starting value for the accumulators, must
     * be >= 0.0.
     * @param l1Strength L1 Regularization Strength
     * @param l2Strength L2 Regularization Strength
     */
    private void initConfig(float learningRate, float initialAccumulatorValue, float l1Strength, float l2Strength) {
        this.learningRate = learningRate;
        config.put(NAME_KEY, this.getOptimizerName());
        config.put(LEARNING_RATE_KEY, learningRate);
        config.put(INITIAL_ACCUM_KEY, initialAccumulatorValue);
        config.put(L1STRENGTH_KEY, l1Strength);
        config.put(L2STRENGTH_KEY, l2Strength);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Map<String, Object> getConfig() {
        return config;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public float getLearningRate() {
        return this.learningRate;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void setLearningRate(float learningRate) {
        this.learningRate = learningRate;
    }

}
