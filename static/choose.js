$(document).ready(function () {
    $('#is_existing').change(function () {
        if (this.checked) {

            $("#newfiles").hide();
            $("#existingfiles").show();
        }
        else {
            // $("#existingfiles").find(":input").prop("disabled", true);
            // $("#newfiles").find(":input").prop("disabled", false);
            $("#newfiles").show();
            $("#existingfiles").hide();
        }
    });

    // $("#existingfiles").find(":input").prop("disabled", true);
    $("#existingfiles").hide();

})
;

