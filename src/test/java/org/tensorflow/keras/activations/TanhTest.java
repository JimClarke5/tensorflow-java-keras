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
package org.tensorflow.keras.activations;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import org.tensorflow.EagerSession;
import org.tensorflow.Operand;
import org.tensorflow.keras.utils.PrintUtils;
import org.tensorflow.op.Ops;
import org.tensorflow.tools.buffer.DataBuffers;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TFloat64;
import org.tensorflow.types.TInt32;

/**
 *
 * @author Jim Clarke
 */
public class TanhTest {
    
    private static final double EPSILON = 1e-7;
    private static final float EPSILON_F = 1e-7f;
    
    
    public TanhTest() {
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
     * Test of Tanh call method.
     */
    @Test
    public void testCall_Int() {
        System.out.println("Tanh int");
        int[] input = {1, -2, 3, -4, -1, 2, -3, 4};
        int[] actual = { 0, 0, 0, 0, 0, 0, 0, 0};
        int[] expected = { };
        try (EagerSession session = EagerSession.create()) {
           Ops tf = Ops.create(session);
           Tanh<TInt32> instance = new Tanh<>();
            Operand<TInt32> operand = instance.call(tf, tf.constant(input));
            operand.asTensor().data().read(DataBuffers.of(actual));
            PrintUtils.printTInt32(operand.asTensor());
            assertArrayEquals(actual,expected);
            fail();
        }catch(AssertionError ex) {
            // expected
            //fail(ex);
        }
    }

    /**
     * Test of Tanh call method.
     */
    @Test
    public void testCall__Float() {
        System.out.println("Tanh float");
        float[] input = {1,2,3,4,5,6,7,8};
        float[] actual = new float[input.length];
        float[] expected = {
            0.7615942F, 0.9640276F, 0.9950547F, 0.9993292F, 0.99990916F, 0.99998784F, 0.99999833F, 1.0F};
        try (EagerSession session = EagerSession.create()) {
           Ops tf = Ops.create(session);
           Tanh<TFloat32> instance = new Tanh<>();
            Operand<TFloat32> operand = instance.call(tf, tf.constant(input));
            operand.asTensor().data().read(DataBuffers.of(actual));
            PrintUtils.print(operand.asTensor());
            assertArrayEquals(expected,actual, EPSILON_F);
        }
    }
    
    /**
     * Test of Tanh call method.
     */
    @Test
    public void testCall__Double() {
        System.out.println("Softmax double");
        double[] input = {1,2,3,4,5,6,7,8};
        double[] actual = { 0, 0, 0, 0, 0, 0, 0, 0};
        double[] expected = {
           0.7615941559557649, 0.9640275800758169, 0.9950547536867305, 
            0.999329299739067, 0.9999092042625951, 0.9999877116507956, 
            0.9999983369439447, 0.9999997749296758 };
        try (EagerSession session = EagerSession.create()) {
           Ops tf = Ops.create(session);
           Tanh<TFloat64> instance = new Tanh<>();
            Operand<TFloat64> operand = instance.call(tf, tf.constant(input));
            operand.asTensor().data().read(DataBuffers.of(actual));
            PrintUtils.print(operand.asTensor());
            assertArrayEquals(expected, actual, EPSILON);
        }
    }
    
    
}
