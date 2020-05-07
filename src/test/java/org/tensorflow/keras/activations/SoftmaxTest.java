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
public class SoftmaxTest {
    
    private static final double EPSILON = 1e-7;
    private static final float EPSILON_F = 1e-7f;
    
    public SoftmaxTest() {
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
     * Test of Softmax method, of class Activations.
     */
    @Test
    public void testRelu_Ops_Operand_Int() {
        System.out.println("Softmax int");
        int[] input = {1, -2, 3, -4, -1, 2, -3, 4};
        int[] actual = { 0, 0, 0, 0, 0, 0, 0, 0};
        int[] expected = { };
        try (EagerSession session = EagerSession.create()) {
           Ops tf = Ops.create(session);
           Softmax<TInt32> instance = new Softmax<>();
            Operand<TInt32> operand = instance.call(tf, tf.constant(input));
            operand.asTensor().data().read(DataBuffers.of(actual));
            PrintUtils.printTInt32(operand.asTensor());
            assertArrayEquals(expected, actual);
            fail();
        }catch(AssertionError ex) {
            // expected
            //fail(ex);
        }
    }

    /**
     * Test of Softmax method, of class Activations.
     */
    @Test
    public void testRelu_Ops_Operand_Float() {
        System.out.println("Softmax float");
        float[] input = {1,2,3,4,5,6,7,8};
        float[] actual = new float[input.length];
        float[] expected = {
            0.07550783F, 0.20525156F, 0.5579316F, 1.5166154F, 4.1225877F, 11.206356F, 30.462032F, 82.80439F};
        try (EagerSession session = EagerSession.create()) {
           Ops tf = Ops.create(session);
           Softmax<TFloat32> instance = new Softmax<>();
            Operand<TFloat32> operand = instance.call(tf, tf.constant(input));
            operand.asTensor().data().read(DataBuffers.of(actual));
            PrintUtils.print(operand.asTensor());
            assertArrayEquals(expected,actual, EPSILON_F);
        }
    }
    
    /**
     * Test of Softmax method, of class Activations.
     */
    @Test
    public void testRelu_Ops_Operand_Double() {
        System.out.println("Softmax double");
        double[] input = {1,2,3,4,5,6,7,8};
        double[] actual = { 0, 0, 0, 0, 0, 0, 0, 0};
        double[] expected = {
           0.07550782856830682, 0.20525155830362918, 0.5579315811996575, 
            1.516615278698451, 4.12258775284935, 11.206355374798198, 
            30.462032178568293, 82.8043885289369   };
        try (EagerSession session = EagerSession.create()) {
           Ops tf = Ops.create(session);
           Softmax<TFloat64> instance = new Softmax<>();
            Operand<TFloat64> operand = instance.call(tf, tf.constant(input));
            operand.asTensor().data().read(DataBuffers.of(actual));
            PrintUtils.print(operand.asTensor());
            assertArrayEquals(expected, actual, EPSILON);
        }
    }
    
}
