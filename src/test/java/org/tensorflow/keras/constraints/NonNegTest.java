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
package org.tensorflow.keras.constraints;

import java.util.Random;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.tensorflow.Operand;
import org.tensorflow.keras.utils.TestSession;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.types.TFloat32;

/**
 *
 * @author jbclarke
 */
public class NonNegTest {
     private TestSession.Mode tf_mode = TestSession.Mode.GRAPH;
    
    public NonNegTest() {
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
    
    private float[] getSampleArray() {
        Random rand = new Random(3537l);
        float[] result = new float[100 * 100];
        for(int i = 0; i < result.length; i++) {
            result[i] = rand.nextFloat() * 100 - 50;
        }
        result[0] = 0;
        return result;
    }

    /**
     * Test of call method, of class NonNeg.
     */
    @Test
    public void testCall() {
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            final float[] array = getSampleArray();
            Operand<TFloat32> weights = tf.reshape(tf.constant(array), tf.constant(Shape.of(100,100)));
            NonNeg instance = new NonNeg(tf);
            Operand result = instance.call(weights);
            session.evaluate(result, (Number v) ->  v.floatValue() >= 0.0f);
        }
    }
    
}
