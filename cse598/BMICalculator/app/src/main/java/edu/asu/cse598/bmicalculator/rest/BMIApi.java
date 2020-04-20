package edu.asu.cse598.bmicalculator.rest;

import retrofit2.Call;
import retrofit2.http.GET;
import retrofit2.http.Query;

public interface BMIApi {
    @GET("calculateBMI")
    Call<BMIResult> calculateBmi(@Query("height") String height, @Query("weight") String weight);
}
