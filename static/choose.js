$(document).ready(function () {
    $('#old_or_new').change(function () {
        if (this.checked) {
            $("#newfiles :input").prop("disabled", true);
            $("#existingfiles :input").prop("disabled", false);
        }
        else {
            $("#existingfiles :input").prop("disabled", true);
            $("#newfiles :input").prop("disabled", false);
        }
    });

    $("#existingfiles :input").prop("disabled", true);

})
;
