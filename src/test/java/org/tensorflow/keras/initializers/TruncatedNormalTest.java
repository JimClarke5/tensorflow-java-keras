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
 * Test the TruncatedNormal initializer
 */
public class TruncatedNormalTest {

    private TestSession.Mode tf_mode = TestSession.Mode.EAGER;

    private static final long SEED = 1000L;
    private static final double MEAN_VALUE = 0.0;
    private static final double STDDEV_VALUE = 3.0;

    public TruncatedNormalTest() {
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
     * Test of getConfig method, of class TruncatedNormal.
     */
    @Test
    public void testGetConfig() {
        Map<String, Object> config = new HashMap<>();
        config.put(TruncatedNormal.MEAN_KEY, MEAN_VALUE);
        config.put(TruncatedNormal.STDDEV_KEY, STDDEV_VALUE);
        config.put(TruncatedNormal.SEED_KEY, SEED);
        TruncatedNormal instance = new TruncatedNormal(null, MEAN_VALUE, STDDEV_VALUE, SEED);
        Map<String, Object> expResult = config;
        Map<String, Object> result = instance.getConfig();
        assertEquals(expResult, result);
    }

    /**
     * Test of getConfig method, of class TruncatedNormal.
     */
    @Test
    public void testConfigCTORMap() {
        Map<String, Object> config = new HashMap<>();
        config.put(TruncatedNormal.MEAN_KEY, MEAN_VALUE);
        config.put(TruncatedNormal.STDDEV_KEY, STDDEV_VALUE);
        config.put(TruncatedNormal.SEED_KEY, SEED);
        TruncatedNormal instance = new TruncatedNormal(null, config);
        Map<String, Object> expResult = config;
        Map<String, Object> result = instance.getConfig();
        assertEquals(expResult, result);
    }

    /**
     * Test of call method, of class TruncatedNormal.
     */
    @Test
    public void testCall_Float() {
        float[] expected = {-1.955122F, -1.0945456F, -0.29379985F, -1.1886811F};
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Shape shape = Shape.of(2, 2);
            TruncatedNormal<TFloat32> instance
                    = new TruncatedNormal(tf, MEAN_VALUE, STDDEV_VALUE, SEED);
            Operand<TFloat32> operand = instance.call(tf.constant(shape), TFloat32.DTYPE);
            session.evaluate(expected, operand);
        }
    }

    @Test
    public void testCall_Double() {
        double[] expected = {5.58717960737721, -4.6606361225803825,
            -2.0188567598844402, 1.8179715736711362};
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Shape shape = Shape.of(2, 2);
            TruncatedNormal<TFloat64> instance
                    = new TruncatedNormal(tf, MEAN_VALUE, STDDEV_VALUE, SEED);
            Operand<TFloat64> operand = instance.call(tf.constant(shape), TFloat64.DTYPE);
            session.evaluate(expected, operand);
        }
    }

}
