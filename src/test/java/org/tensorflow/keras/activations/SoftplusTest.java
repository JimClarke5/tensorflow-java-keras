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
import org.tensorflow.ndarray.buffer.DataBuffers;
import org.tensorflow.op.Ops;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TFloat64;
import org.tensorflow.types.TInt32;

/**
 *
 * @author Jim Clarke
 */
public class SoftplusTest {

    private static final double EPSILON = 1e-7;
    private static final float EPSILON_F = 1e-7f;

    public SoftplusTest() {
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
     * Test of Softplus call method
     */
    @Test
    public void testCall__Int() {
        int[] input = {1, -2, 3, -4, -1, 2, -3, 4};
        int[] actual = {0, 0, 0, 0, 0, 0, 0, 0};
        int[] expected = {};
        try (EagerSession session = EagerSession.create()) {
            Ops tf = Ops.create(session);
            Softplus<TInt32> instance = new Softplus<>(tf);
            Operand<TInt32> operand = instance.call(tf.constant(input));
            operand.asTensor().data().read(DataBuffers.of(actual));
            assertArrayEquals(expected, actual);
            fail();
        } catch (AssertionError ex) {
            // expected
            //fail(ex);
        }
    }

    /**
     * Test of Softplus call method
     */
    @Test
    public void testCall__Float() {
        float[] input = {1, 2, 3, 4, 5, 6, 7, 8};
        float[] actual = new float[input.length];
        float[] expected = {
            1.3132616F, 2.126928F, 3.0485873F, 4.01815F, 5.0067153F, 6.0024757F, 7.0009117F, 8.000336F};
        try (EagerSession session = EagerSession.create()) {
            Ops tf = Ops.create(session);
            Softplus<TFloat32> instance = new Softplus<>(tf);
            Operand<TFloat32> operand = instance.call(tf.constant(input));
            operand.asTensor().data().read(DataBuffers.of(actual));
            assertArrayEquals(expected, actual, EPSILON_F);
        }
    }

    /**
     * Test of Softplus call method
     */
    @Test
    public void testCall__Double() {
        double[] input = {1, 2, 3, 4, 5, 6, 7, 8};
        double[] actual = {0, 0, 0, 0, 0, 0, 0, 0};
        double[] expected = {
            1.3132616875182228, 2.1269280110429727, 3.048587351573742,
            4.0181499279178094, 5.006715348489118, 6.00247568513773,
            7.000911466453774, 8.000335406372896,};
        try (EagerSession session = EagerSession.create()) {
            Ops tf = Ops.create(session);
            Softplus<TFloat64> instance = new Softplus<>(tf);
            Operand<TFloat64> operand = instance.call(tf.constant(input));
            operand.asTensor().data().read(DataBuffers.of(actual));
            assertArrayEquals(expected, actual, EPSILON);
        }
    }

}
