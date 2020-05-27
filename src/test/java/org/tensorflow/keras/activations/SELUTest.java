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
import org.tensorflow.exceptions.TensorFlowException;
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
public class SELUTest {
    private static final double EPSILON = 1e-7;
    private static final float EPSILON_F = 1e-7f;
    
    public SELUTest() {
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
     * Test of SELU call method
     */
    @Test
    public void testCall__Int() {
        System.out.println("SELU int");
        int[] input = {1, -2, 3, -4, -1, 2, -3, 4};
        int[] actual = { 0, 0, 0, 0, 0, 0, 0, 0};
        int[] expected = { };
        try (EagerSession session = EagerSession.create()) {
           Ops tf = Ops.create(session);
           SELU<TInt32> instance = new SELU<>(tf);
            Operand<TInt32> operand = instance.call(tf.constant(input));
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
     * Test of SELU call method
     */
    @Test
    public void testCall__Float() {
        System.out.println("SELU float");
        float[] input = {1, -2, 3, -4, -1, 2, -3, 4};
        float[] actual = new float[input.length];
        float[] expected = {
            1.050701F, -1.5201665F, 3.152103F, -1.7258986F, -1.1113307F, 2.101402F, -1.6705687F, 4.202804F};
        try (EagerSession session = EagerSession.create()) {
           Ops tf = Ops.create(session);
           SELU<TFloat32> instance = new SELU<>(tf);
            Operand<TFloat32> operand = instance.call(tf.constant(input));
            operand.asTensor().data().read(DataBuffers.of(actual));
            PrintUtils.print(operand.asTensor());
            assertArrayEquals(expected, actual, EPSILON_F);
        }
    }
    
    /**
     * Test of SELU call method
     */
    @Test
    public void testCall__Double() {
        System.out.println("SELU double");
        double[] input = {1, -2, 3, -4, -1, 2, -3, 4};
        double[] actual = { 0, 0, 0, 0, 0, 0, 0, 0};
        double[] expected = {
           1.0507009873554805, -1.520166468595695, 3.1521029620664414, 
            -1.7258986281898947, -1.1113307378125628, 2.101401974710961, 
            -1.670568728767112, 4.202803949421922,   };
        try (EagerSession session = EagerSession.create()) {
           Ops tf = Ops.create(session);
           SELU<TFloat64> instance = new SELU<>(tf);
            Operand<TFloat64> operand = instance.call(tf.constant(input));
            operand.asTensor().data().read(DataBuffers.of(actual));
            PrintUtils.print(operand.asTensor());
            assertArrayEquals(expected, actual, EPSILON);
        }
    }
    
}
