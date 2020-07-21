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
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import org.tensorflow.Operand;
import org.tensorflow.Tensor;
import org.tensorflow.keras.utils.ND;
import org.tensorflow.keras.utils.TestSession;
import org.tensorflow.op.Ops;
import org.tensorflow.tools.Shape;
import org.tensorflow.tools.ndarray.FloatNdArray;
import org.tensorflow.tools.ndarray.NdArrays;
import org.tensorflow.types.TFloat32;

/**
 *
 * @author jbclarke
 */
public class MinMaxNormTest {
    
     private TestSession.Mode tf_mode = TestSession.Mode.GRAPH;
    
    public MinMaxNormTest() {
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
     * Test of getConfig method, of class Constant.
     */
    @Test
    public void testConfig() {
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            MinMaxNorm instance = new MinMaxNorm(tf, 0.5f, 1.5f, 0.9f, 1);
            
            assertEquals(0.5f, instance.getMinValue());
            assertEquals(1.5f, instance.getMaxValue());
            assertEquals(0.9f, instance.getRate());
            assertEquals(1, instance.getAxis());
        }
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
     * Test of call method, of class MinMaxNorm.
     */
    @Test
    public void testCall() {
        float[] testValues = {0.1f, 0.5f, 3f, 8f, 1e-7f};
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            final float[] array = getSampleArray();
            Operand<TFloat32> weights = tf.reshape(tf.constant(array), tf.constant(Shape.of(100,100)));
            for(AtomicInteger i = new AtomicInteger(); i.get() < testValues.length; i.getAndIncrement() ) {
                MinMaxNorm instance = new MinMaxNorm(tf, testValues[i.get()], testValues[i.get()]*2);
                Operand result = instance.call(weights);
                try (Tensor<TFloat32> tensor = session.getGraphSession().runner().fetch(result).run().get(0).expect(TFloat32.DTYPE)) {
                    evaluate(session, tensor, testValues[i.get()]);
                }
            }
        }
    }
    
    private void evaluate(TestSession session, Tensor<TFloat32> tensor, float m) {
        FloatNdArray tensorArray = NdArrays.ofFloats(tensor.shape());
        tensor.data().copyTo(tensorArray);
        tensorArray = ND.square(tensorArray);
        FloatNdArray normArray = ND.sum(tensorArray, 0);
        FloatNdArray normOfNormalized = ND.sqrt(normArray);
        session.evaluate(normOfNormalized, (f) -> f.floatValue() >= m && f.floatValue() <=  m * 2f + 1e-5f);
    }
    
}
