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
package org.tensorflow.keras.initializers;

import java.util.HashMap;
import java.util.Map;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import org.tensorflow.Operand;
import org.tensorflow.keras.utils.TestSession;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TFloat64;

/**
 * Test cases for GlorotUniform initializer
 */
public class GlorotUniformTest {

    private TestSession.Mode tf_mode = TestSession.Mode.EAGER;

    private static final long SEED = 1000L;

    int counter;

    public GlorotUniformTest() {
    }

    @BeforeAll
    public static void setUpClass() {
    }

    @AfterAll
    public static void tearDownClass() {
    }

    @BeforeEach
    public void setUp() {
    }

    @AfterEach
    public void tearDown() {
    }

    /**
     * Test of getConfig method, of class GlorotUniform.
     */
    @Test
    public void testGetConfig() {
        Map<String, Object> config = new HashMap<>();
        config.put(GlorotUniform.SCALE_KEY, 1.0);
        config.put(GlorotUniform.MODE_KEY, "fan_avg");
        config.put(GlorotUniform.DISTRIBUTION_KEY, "uniform");
        config.put(GlorotUniform.SEED_KEY, SEED);
        GlorotUniform instance = new GlorotUniform(null, SEED);
        Map<String, Object> expResult = config;
        Map<String, Object> result = instance.getConfig();
        assertEquals(expResult, result);
    }

    /**
     * Test of getConfig method, of class GlorotUniform.
     */
    @Test
    public void testConfigCTORMap() {
        Map<String, Object> config = new HashMap<>();
        config.put(GlorotUniform.SCALE_KEY, 1.0);
        config.put(GlorotUniform.MODE_KEY, "fan_avg");
        config.put(GlorotUniform.DISTRIBUTION_KEY, "uniform");
        config.put(GlorotUniform.SEED_KEY, SEED);
        GlorotUniform instance = new GlorotUniform(null, config);
        Map<String, Object> expResult = config;
        Map<String, Object> result = instance.getConfig();
        assertEquals(expResult, result);
    }

    /**
     * Test of call method, of class GlorotUniform.
     */
    @Test
    public void testCall_Float() {
        float[] expected = {0.9266439F, 0.8190767F, 1.1268647F, 0.6596042F};
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Shape shape = Shape.of(2, 2);
            GlorotUniform<TFloat32> instance
                    = new GlorotUniform<>(tf, SEED);
            Operand<TFloat32> operand = instance.call(tf.constant(shape), TFloat32.DTYPE);
            session.evaluate(expected, operand);
        }
    }

    @Test
    public void testCall_Double() {
        double[] expected = {0.06468193804916589, 0.44170328686673477,
            0.06711059208157763, 0.6278720842445181};
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Shape shape = Shape.of(2, 2);
            GlorotUniform<TFloat64> instance
                    = new GlorotUniform<>(tf, SEED);
            Operand<TFloat64> operand = instance.call(tf.constant(shape), TFloat64.DTYPE);
            session.evaluate(expected, operand);
        }
    }

}
