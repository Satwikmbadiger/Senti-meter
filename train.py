import models.logistic as model_1
import models.naives as model_2
import models.mdl3 as model_3
try:
    model_1.main()
    model_2.main()
    model_3.main()

except Exception as e:
    print(f"Error :  {e}")
