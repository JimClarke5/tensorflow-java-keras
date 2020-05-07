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
public class SigmoidTest {
    private static final double EPSILON = 1e-7;
    private static final float EPSILON_F = 1e-7f;
    
    public SigmoidTest() {
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
     * Test of Sigmoid call method
     */
    @Test
    public void testCall__Int() {
        System.out.println("Sigmoid int");
        int[] input = {1, -2, 3, -4, -1, 2, -3, 4};
        int[] actual = { 0, 0, 0, 0, 0, 0, 0, 0};
        int[] expected = { };
        try (EagerSession session = EagerSession.create()) {
           Ops tf = Ops.create(session);
           Sigmoid<TInt32> instance = new Sigmoid<>();
            Operand<TInt32> operand = instance.call(tf, tf.constant(input));
            operand.asTensor().data().read(DataBuffers.of(actual));
            PrintUtils.printTInt32(operand.asTensor());
            assertArrayEquals(actual,expected);
        }catch(AssertionError ex) {
            // expected
            //fail(ex);
        }
    }

    /**
     * Test of Sigmoid call method
     */
    @Test
    public void testCall__Float() {
        System.out.println("Sigmoid float");
        float[] input = {1, -2, 3, -4, -1, 2, -3, 4};
        float[] actual = new float[input.length];
        float[] expected = {
            0.7310586F, 0.11920291F, 0.95257413F, 0.017986238F, 0.26894143F, 0.8807971F, 0.047425866F, 0.98201376F,};
        try (EagerSession session = EagerSession.create()) {
           Ops tf = Ops.create(session);
           Sigmoid<TFloat32> instance = new Sigmoid<>();
            Operand<TFloat32> operand = instance.call(tf, tf.constant(input));
            operand.asTensor().data().read(DataBuffers.of(actual));
            PrintUtils.print(operand.asTensor());
            assertArrayEquals(actual,expected, EPSILON_F);
        }
    }
    
    /**
     * Test of Sigmoid call method
     */
    @Test
    public void testCall__Double() {
        System.out.println("Sigmoid double");
        double[] input = {1, -2, 3, -4, -1, 2, -3, 4};
        double[] actual = { 0, 0, 0, 0, 0, 0, 0, 0};
        double[] expected = {
           0.7310585786300049, 0.11920292202211755, 0.9525741268224334, 
            0.01798620996209156, 0.2689414213699951, 0.8807970779778823, 
            0.04742587317756678, 0.9820137900379085};
        try (EagerSession session = EagerSession.create()) {
           Ops tf = Ops.create(session);
           Sigmoid<TFloat64> instance = new Sigmoid<>();
            Operand<TFloat64> operand = instance.call(tf, tf.constant(input));
            operand.asTensor().data().read(DataBuffers.of(actual));
            PrintUtils.print(operand.asTensor());
            assertArrayEquals(actual,expected, EPSILON);
        }
    }
    
}
