package edu.asu.cse598.bmicalculator;

import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.EditText;

import androidx.appcompat.app.AppCompatActivity;

import java.util.List;

import edu.asu.cse598.bmicalculator.rest.BMIApi;
import edu.asu.cse598.bmicalculator.rest.BMIApiClient;
import edu.asu.cse598.bmicalculator.rest.BMIResult;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class MainActivity extends AppCompatActivity {

    public static final String EXTRA_MESSAGE = "edu.asu.cse598.bmicalculator.MESSAGE";
    private static final String TAG = MainActivity.class.getSimpleName();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
    }

    /**
     * Called when the user taps the Send button
     */
    public void sendMessage(View view) {
        BMIApi bmiService = BMIApiClient.getClient().create(BMIApi.class);
        Call<BMIResult> call = bmiService.calculateBmi("60", "156");
        call.enqueue(new Callback<BMIResult>() {
            @Override
            public void onResponse(Call<BMIResult> call, Response<BMIResult> response) {

                String bmi = response.body().getBmi();
                List<String> more = response.body().getMore();
                String risk = response.body().getRisk();

                Log.i(TAG, ">>> " + bmi);
                Log.i(TAG, ">>> " + more.toString());
                Log.i(TAG, ">>> " + risk);

                ((EditText) findViewById(R.id.editText)).append(more.get(0));

                // TODO open the first link from "more" array in browser
//                Intent browserIntent = new Intent(Intent.ACTION_VIEW, Uri.parse(more.get(0)));
//                startActivity(browserIntent);
            }

            @Override
            public void onFailure(Call<BMIResult> call, Throwable t) {
                Log.e(TAG, t.toString());
            }
        });
    }
}
